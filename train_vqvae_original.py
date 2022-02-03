import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
from dataset import CelebaDataset

import face_alignment


def train(epoch, loader, model, fa_model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    landmark_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    lm_sum = 0

    # for i, (img, label) in enumerate(loader):
    for i, (source, target) in enumerate(loader):
        model.zero_grad()

        # img = img.to(device)
        source, target = source.to(device), target.to(device)

        out, latent_loss = model(source)

        with torch.no_grad():
            out_landmarks = fa_model.get_landmarks_from_batch(out)
            target_landmarks = fa_model.get_landmarks_from_batch(target)

        # check if all faces were detected
        if len(out_landmarks) == out.shape[0] and len(target_landmarks) == target.shape[0]:
            print(f'Lengths - out_landmarks : {len(out_landmarks)}, target_landmarks : {len(target_landmarks)}')
            out_landmarks_torch = torch.tensor(out_landmarks).float()
            target_landmarks_torch = torch.tensor(target_landmarks).float()
            # landmark_loss = criterion(torch.tensor(out_landmarks), torch.tensor(target_landmarks))
            landmark_loss = criterion(out_landmarks_torch, target_landmarks_torch)
        else:
            landmark_loss = 0

        # recon_loss = criterion(out, img)
        recon_loss = criterion(out, target) # Get the recon loss between generate and target img
        
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss  + landmark_loss_weight * landmark_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * source.shape[0]
        part_lm_sum = landmark_loss.item() * source.shape[0]
        part_mse_n = source.shape[0]
        # TODO: faces may not always be detected, adjust lm_n in that case
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n, "lm_sum": part_lm_sum}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]
            lm_sum += part["lm_sum"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}: avg lm loss: {lm_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = source[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([target[:sample_size], out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CelebaDataset(args.path, transform)

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE().to(device)
    # Do we need distributed data parallel for this model? 
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        # fa_model = nn.parallel.DistributedDataParallel(
        #     fa_model,
        #     device_ids=[dist.get_local_rank()],
        #     output_device=dist.get_local_rank(),
        # )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, fa_model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args, flush=True)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
