import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
        in_channels,
        latent_dim,
        hidden_dims = [128, 64],
        kernel_size = 3,
        stride = 2,
        image_size = (28, 28)):
        super().__init__()

        out_height, out_width = image_size
        for _ in range(len(hidden_dims)):
            out_height = (out_height - kernel_size + 2) // stride + 1
            out_width = (out_width - kernel_size + 2) // stride + 1

        flatten_dim = hidden_dims[-1]*out_height*out_width

        dims = [in_channels] + hidden_dims
        conv = []
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            conv.append(nn.Conv2d(dim1, dim2, kernel_size = kernel_size, stride = stride, padding = 1))
            conv.append(nn.ReLU())
        conv.append(nn.Flatten())

        self.conv = nn.Sequential(*conv)
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_var = nn.Linear(flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self,
        in_channels,
        latent_dim,
        hidden_dims = [128, 64],
        kernel_size = 3,
        stride = 2,
        image_size = (28, 28)):
        super().__init__()

        self.image_size = image_size
        self.last_hidden_dim = hidden_dims[-1]
        self.out_height, self.out_width = image_size
        out_padding = []
        for _ in range(len(hidden_dims)):
            out_padding.append(((self.out_height - kernel_size) % 2, (self.out_width - kernel_size) % 2))
            self.out_height = (self.out_height - kernel_size + 2) // stride + 1
            self.out_width = (self.out_width - kernel_size + 2) // stride + 1

        self.fc = nn.Linear(latent_dim, self.last_hidden_dim * self.out_height * self.out_width)
        
        dims = hidden_dims[::-1] + [in_channels]
        conv_t = []
        for dim1, dim2, output_padding in zip(dims[:-1], dims[1:], out_padding[::-1]):
            conv_t.append(nn.ConvTranspose2d(dim1, dim2, kernel_size = kernel_size, stride = stride, padding = 1, output_padding = output_padding))
            conv_t.append(nn.ReLU())
        self.conv_t = nn.Sequential(*conv_t)

        final_output_padding = -(kernel_size - 3) 
        self.output_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size = kernel_size, stride = 1, padding = 1, output_padding = final_output_padding)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.last_hidden_dim, self.out_height, self.out_width)
        x = self.conv_t(x)
        x = self.output_conv(x, output_size=self.image_size)
        return x

class VAE(nn.Module):
    def __init__(self,
        in_channels,
        latent_dim,
        hidden_dims = [128, 64],
        kernel_size = 3,
        stride = 2,
        image_size = (28, 28)):

        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_channels, latent_dim, hidden_dims, kernel_size, stride, image_size)
        self.decoder = Decoder(in_channels, latent_dim, hidden_dims, kernel_size, stride, image_size)

    def reparametrize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        return out

    def train_loss(self, x, loss):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        recon_loss = loss(out, x)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return recon_loss + kl_loss, recon_loss, kl_loss

    def generate(self, size):
        z = torch.randn(size, self.latent_dim)
        out = self.decoder(z)
        out = torch.sigmoid(out)
        return out

if __name__ == '__main__':
    from bandcamp_dataset import bandcamp_dataset
    from tqdm import tqdm
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 64
    train, test = bandcamp_dataset()
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    model = VAE(in_channels=3, latent_dim=8, hidden_dims=[32, 64])

    epochs = 50
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # loop over the dataset multiple times
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        rec_loss = 0
        kl_loss = 0

        for i, data in enumerate(train_loader):
            inputs, _ = data
            inputs = inputs.to(device)
    
            optimizer.zero_grad()
    
            loss, rec, kl = model.train_loss(inputs, criterion)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            rec_loss += rec.item()
            kl_loss += kl.item()

        print(f'epoch {epoch}, loss {running_loss}, rec {rec_loss}, kl {kl_loss}')
    
    breakpoint()