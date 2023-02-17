import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,
        in_channels = 3,
        hidden_dims = [32, 64, 128, 256],
        kernel_size = 4,
        stride = 2,
        image_size = (64, 64)):
        
        super().__init__()

        dims = [in_channels] + hidden_dims
        paddings = [1] * len(dims)
        paddings[-1] = 0

        conv = []
        for dim1, dim2, pad in zip(dims[:-1], dims[1:], paddings):
            conv.append(nn.Conv2d(dim1, dim2, kernel_size = kernel_size, stride = stride, padding = pad, bias = False))
            conv.append(nn.BatchNorm2d(dim2))
            conv.append(nn.LeakyReLU(0.02, inplace=True))
        
        conv.append(nn.Conv2d(dims[-1], 1, kernel_size = kernel_size, stride = stride, padding = paddings[-1], bias = False))
        conv.append(nn.Sigmoid())

        self.net = nn.Sequential(*conv)

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self,
        noise_dim,
        out_channels = 3,
        hidden_dims = [256, 128, 64, 32],
        kernel_size = 4,
        stride = 2,
        image_size = (64, 64)):

        super().__init__()

        dims = [noise_dim] + hidden_dims
        paddings = [1] * len(dims)
        paddings[0] = 0

        conv = []
        for dim1, dim2, pad in zip(dims[:-1], dims[1:], paddings):
            conv.append(nn.ConvTranspose2d(dim1, dim2, kernel_size = kernel_size, stride = stride, padding = pad, bias = False))
            conv.append(nn.BatchNorm2d(dim2))
            conv.append(nn.LeakyReLU())
        
        conv.append(nn.ConvTranspose2d(dims[-1], out_channels, kernel_size = kernel_size, stride = stride, padding = paddings[-1], bias = False))
        conv.append(nn.Sigmoid())

        self.net = nn.Sequential(*conv)

    def forward(self, x):
        return self.net(x)


class DCGAN(nn.Module):
    def __init__(self,
        noise_dim,
        in_channels = 3,
        hidden_dims = [32, 64, 128, 256],
        kernel_size = 4,
        stride = 2,
        image_size = (64, 64),
        device = 'cpu'):

        super().__init__()

        self.noise_dim = noise_dim
        self.img_channels = in_channels
        self.img_size = image_size
        self.device = device
        self.discriminator = Discriminator(in_channels, hidden_dims, kernel_size, stride, image_size).to(self.device)
        self.generator = Generator(noise_dim, in_channels, hidden_dims[::-1], kernel_size, stride, image_size).to(self.device)

    def generate(self, size):
        noise = torch.randn(size, self.noise_dim, 1, 1, device=self.device)
        output = self.generator(noise)
        labels = torch.full((size,), 0., device=self.device)
        return output, labels

    def discriminate(self, input, label):
        output = self.discriminator(input).view(-1)
        labels = torch.full((input.size(0),), label, device=self.device)
        return output, labels
        
    def generate_and_discriminate(self, size):
        generator_output, discriminator_labels = self.generate(size)
        discriminator_output = self.discriminator(generator_output.detach()).view(-1)
        return generator_output, discriminator_output, discriminator_labels

    def train_loop(self, batched_data, batch_size, criterion, optimizer_D, optimizer_G):

        # 1. update discriminator

        self.discriminator.zero_grad()

        # discriminate real data
        d_real_output, label = self.discriminate(batched_data, 1.)
        errD_real = criterion(d_real_output, label)
        errD_real.backward()

        # discriminate fake data
        g_output, d_fake_output, label = self.generate_and_discriminate(batch_size)
        errD_fake = criterion(d_fake_output, label)
        errD_fake.backward()

        # step to update discriminator
        optimizer_D.step()
        errD = errD_real + errD_fake

        # 2. update generator

        self.generator.zero_grad()

        # we can use data generated earlier, and use updated discriminator output
        output, label = self.discriminate(g_output,  1.)
        errG = criterion(output, label)
        errG.backward()
        optimizer_G.step()

        return errD.item(), errG.item()

        
