import torch
import torch.nn as nn

# The generator will be composed of a series of 2d transpose convolutions - upsampling
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_channels=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.make_generator_layer(z_dim, hidden_dim*4),
            self.make_generator_layer(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.make_generator_layer(hidden_dim*2, hidden_dim),
            self.make_generator_layer(hidden_dim, im_channels, kernel_size=4, final_layer=True)
        )
        
    def forward(self, noise):
        # print(f'Noise shape inside the generator : {noise.shape}')
        return self.generator(noise)
        
    def make_generator_layer(self, in_channels, out_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.Tanh() # returns a pixel value between -1 and 1
            )
        
# Sample noise for the generator 
def sample_noise(batch_size=64, z_dim=10, device='cpu'):
    return torch.randn(batch_size, z_dim).to(device)

# The discriminator will be composed of 2d convolution operations - downsampling
# downsampling reduces the dimension and increases the number of channels in the image
class Discriminator(nn.Module):
    def __init__(self, im_channels=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            self.make_disc_layer(im_channels, hidden_dim),
            self.make_disc_layer(hidden_dim, hidden_dim*2),
            self.make_disc_layer(hidden_dim*2, 1, final_layer=True)
        )
        
    def forward(self, image):
        disc_pred = self.discriminator(image)
        return disc_pred.view(len(disc_pred), -1)
        
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