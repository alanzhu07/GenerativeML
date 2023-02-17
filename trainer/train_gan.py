import torch
from torch import nn
from torch import optim
from gan.gan_torch import DCGAN

from bandcamp_dataset import bandcamp_dataset
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 64
train, test = bandcamp_dataset(output_size=(64, 64))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

model = DCGAN(64, device=device)

epochs = 50
lr = 1e-3
optimizerD = optim.Adam(model.discriminator.parameters(), lr=lr)
optimizerG = optim.Adam(model.generator.parameters(), lr=lr)
criterion = nn.BCELoss()

# loop over the dataset multiple times
for epoch in tqdm(range(epochs)):
    d_loss = 0
    g_loss = 0

    for i, data in enumerate(train_loader):
        inputs, _ = data
        inputs = inputs.to(device)

        lossD, lossG = model.train_loop(inputs, batch_size, criterion, optimizerD, optimizerG)

        d_loss += lossD
        g_loss += lossG

    print(f'epoch {epoch}, loss {lossD+lossG}, lossD {lossD}, lossG {lossG}')

breakpoint()