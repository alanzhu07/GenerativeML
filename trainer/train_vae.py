import torch
from torch import nn
from torch import optim
from vae.vae_torch import VAE

from bandcamp_dataset import bandcamp_dataset
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 64
train, test = bandcamp_dataset()
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

model = VAE(in_channels=3, latent_dim=64, hidden_dims=[32, 64, 128, 256])

epochs = 50
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

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