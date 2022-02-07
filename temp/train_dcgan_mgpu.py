# MultiGPU implementation of dcgan - will be used for training on the CelebaAHQ dataset
# Deep convolutional generative adversarial network 
# Make this code multi GPU trainable
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm.auto import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

import distributed as dist

from dcgan import Generator, Discriminator, sample_noise

CHECKPOINT_DIR = 'dcgan_checkpoints'
SAMPLE_DIR = 'dcgan_samples'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def train(gen, disc, gen_opt, disc_opt, loader, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.BCEWithLogitsLoss()

    sample_size = 10

    gen_sum = 0
    disc_sum = 0
    total_n = 0

    z_dim = 64

    # Run the training loop on the corresponding GPU
    for index, (real, label) in enumerate(loader):
        loader = tqdm(loader)

        # generate the discriminator classification on the real and fake images
        batch_size = real.shape[0]
        disc_opt.zero_grad()
        real = real.to(device)
        disc_real_pred = disc(real)
        noise = sample_noise(batch_size, z_dim, device)
        noise = noise.view(batch_size, z_dim, 1, 1)
        fake = gen(noise) # dimension -> batch_size x im_channels x height x width
        
        disc_fake_pred = disc(fake.detach()) # gradients shouldn't backpropagate to the generator
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss)/2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()
        
        gen_opt.zero_grad()
        noise = sample_noise(batch_size, z_dim, device)
        noise = noise.view(batch_size, z_dim, 1, 1)
        fake = gen(noise)
        disc_fake_pred = disc(fake)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()
        
        # mean_disc_loss += disc_loss.item() / step_size
        # mean_gen_loss += gen_loss.item() / step_size

        # compute the part generator and discriminator losses - the losses will be averaged for all the batches across all GPUs
        part_gen_loss = fake.shape[0] * gen_loss.item()
        part_disc_loss = real.shape[0] * disc_los.item()
        # the losses will be aggregated across all the GPUs
        comm = {"gen_loss" : part_gen_loss, "disc_loss" : part_disc_loss, "part_n" : real.shape[0]}
        comm = dist.all_gather(comm)

        for part in comm:
            gen_sum += comm['gen_loss']
            disc_sum += comm['disc_loss']
            total_n += comm['part_n']
        
        # if current_step % step_size == 0:
        #     print(f'Epoch : {epoch}, step : {current_step}, mean gen loss : {mean_gen_loss}, mean disc loss : {mean_disc_loss}')
        #     mean_disc_loss = 0
        #     mean_gen_loss = 0

        if dist.is_primary():
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; gen_loss: {gen_loss.item():.3f}; "
                    f"disc_loss: {disc_loss.item():.3f}; avg gen loss: {gen_sum / total_n:.5f}; "
                    f"avg disc loss: {disc_sum / total_n:.5f}"
                )
            )

        # save the images every nth steps
        if i%100 == 0:
            fake_sampled = fake[:sample_size]
            real_sampled = real[:sample_size]

            utils.save_image(
                torch.cat([fake_sampled, real_sampled], 0),
                f"{SAMPLE_DIR}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
            range=(-1, 1),
            )


# This process will be created for every GPU available
def main():
    n_gpu = 2
    device = "cuda"

    distributed = dist.get_world_size() > 1

    # transformation that needs to be applied on the input image 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # this transforms the data between -1 and 1
    ])

    BATCH_SIZE = 64

    dataset = torchvision.datasets.MNIST(root='.', download=False, transform=transform)

    sampler = dist.data_sampler(dataset, shuffle=True, distributed=distributed)
    loader = DataLoader(
        dataset, batch_size=128 // n_gpu, sampler=sampler, num_workers=2
    )

    z_dim = 64
    beta_1 = 0.5
    beta_2 = 0.999
    lr = 2e-4
    epochs = 100

    # binary classification loss for fake and real image
    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)


    # You initialize the weights to the normal distribution
    # with mean 0 and standard deviation 0.02
    # Weight initialization is really important
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # The model is replicated across the GPUs
    if distributed:
        gen = nn.parallel.DistributedDataParallel(
            gen,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        disc = nn.parallel.DistributedDataParallel(
            disc,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # the batches in every epoch will be divided between the GPUs available
    for i in range(epochs):
        train(gen, disc, gen_opt, disc_opt, loader, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/vqvae_{str(i + 1).zfill(3)}.pt")    



if __name__ == '__main__':

    n_gpu = 2

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )

    dist_url = f"tcp://127.0.0.1:{port}"

    dist.launch(main, n_gpu, 1, 0, dist_url)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

    
    # show_tensor_images(real)
    # show_tensor_images(fake)