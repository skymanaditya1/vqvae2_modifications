import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.optim as optim

from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt

import cv2

BATCH_SIZE = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Use the CelebaHQ dataset 
from torch.utils.data import Dataset
class CelebaDataset(Dataset):
    def __init__(self, path, transform):
        self.images = glob(path + '/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        return self.transform(image)

dataset_path = '/ssd_scratch/cvit/aditya1/CelebA-HQ-img/'
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

dataset = CelebaDataset(dataset_path, transform)
loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=1)


def show_tensor_images(image_tensor, num_images=25, size=(3, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

# The generator will be composed of a series of 2d transpose convolutions - upsampling
class Generator(nn.Module):
    def __init__(self, z_dim=64, im_channels=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.make_generator_layer(z_dim, hidden_dim*16),
            self.make_generator_layer(hidden_dim*16, hidden_dim*8),
            self.make_generator_layer(hidden_dim*8, hidden_dim*4, output_padding=0),
            
            self.make_unet_layer(hidden_dim*4, hidden_dim*2),
            self.make_unet_layer(hidden_dim*2, hidden_dim),
            self.make_unet_layer(hidden_dim, im_channels, padding=3, final_layer=True)
        )
        
    def forward(self, noise):
        # print(f'Noise shape inside the generator : {noise.shape}')
        return self.generator(noise)
        
    def make_unet_layer(self, in_channels, out_channels, padding=2, kernel_size=3, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.Tanh()
            )
        
    def make_generator_layer(self, in_channels, out_channels, kernel_size=4, stride=2, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
def sample_noise(batch_size=BATCH_SIZE, z_dim=64, device=device):
        return torch.randn(batch_size, z_dim).to(device)


# The discriminator will be composed of 2d convolution operations - downsampling
# downsampling reduces the dimension and increases the number of channels in the image
class Discriminator(nn.Module):
    def __init__(self, im_channels=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            self.make_conv_layer(im_channels, hidden_dim),
            self.make_conv_layer(hidden_dim, hidden_dim*2),
            self.make_conv_layer(hidden_dim*2, hidden_dim*4),
            
            self.make_disc_layer(hidden_dim*4, hidden_dim*8),
            self.make_disc_layer(hidden_dim*8, hidden_dim*16),
            self.make_disc_layer(hidden_dim*16, 1, final_layer=True)
        )
        
    def forward(self, image):
        disc_pred = self.discriminator(image)
        return disc_pred.view(len(disc_pred), -1)
#         return self.discriminator(image)
        
    def make_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
    def make_disc_layer(self, in_channels, out_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True) # saves memory doing it inplace, doesn't generate a new output
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride) # no activation required for the final layer
            )


z_dim = 64
beta_1 = 0.5
beta_2 = 0.999
lr = 2e-4
epochs = 200

# binary classification loss for fake and real image
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc_opt = optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

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


mean_gen_loss = 0
mean_disc_loss = 0
step_size = 100
current_step = 0
eval_step_size = 200
for epoch in range(epochs):
    for index, real in enumerate(loader):
        # generate the discriminator classification on the real and fake images
        gen.train()
        current_step += 1
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
        
        mean_disc_loss += disc_loss.item() / step_size
        mean_gen_loss += gen_loss.item() / step_size
        
        if current_step % step_size == 0:
            print(f'Epoch : {epoch}, step : {current_step}, mean gen loss : {mean_gen_loss}, mean disc loss : {mean_disc_loss}', flush=True)
            mean_disc_loss = 0
            mean_gen_loss = 0
                
            # show_tensor_images(real)
            # show_tensor_images(fake)