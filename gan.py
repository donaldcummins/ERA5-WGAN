import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# sample from ERA5
df = pd.read_pickle('~/Reanalysis/era5.pkl')

# remove unreliable temperature extrapolations
df = df[df.theta <= 273.15 + 40.0]

# take N rows at random
N = 1000
df = df.sample(n=N, random_state=1)

# construct kinematic fluxes
df['ustar2'] = df.ustar**2
df['ustarthetastar'] = df.ustar*df.thetastar
df['ustarqstar'] = df.ustar*df.qstar

# build data matrix
variables = [
    'zu','zt','u','theta','thetas','q','qs',
    'ustar2','ustarthetastar','ustarqstar'
]
real_data_original = df[variables]

# standardize data
scaler = StandardScaler()
real_data_scaled = scaler.fit_transform(real_data_original)

# generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# hyperparameters
input_dim = real_data_scaled.shape[1]
output_dim = real_data_scaled.shape[1]
lr = 0.00005
batch_size = 64
num_epochs = 1000
n_critic = 5
clip_value = 0.01

# initialize models
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(input_dim)
optimizer_g = optim.RMSprop(generator.parameters(), lr)
optimizer_d = optim.RMSprop(discriminator.parameters(), lr)

# create PyTorch data tensor
real_data_scaled_tensor = torch.FloatTensor(real_data_scaled)

# data loader
data_loader = DataLoader(TensorDataset(real_data_scaled_tensor), batch_size=batch_size, shuffle=True)

# clip discriminator weights
def clip_weights(model, clip_value):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

# training loop
wasserstein_distances = []
generator_losses = []
for epoch in range(num_epochs):
    for batch in data_loader:
        real_data = batch[0]

        # train discriminator
        for _ in range(n_critic):
            optimizer_d.zero_grad()

            fake_data = generator(torch.randn(real_data.size(0), input_dim))
            outputs_fake = discriminator(fake_data.detach())
            outputs_real = discriminator(real_data)

            wasserstein_distance = -torch.mean(outputs_real) + torch.mean(outputs_fake)
            wasserstein_distance.backward()
            optimizer_d.step()

            clip_weights(discriminator, clip_value=0.01)
        
        # train generator
        optimizer_g.zero_grad()

        fake_data = generator(torch.randn(real_data.size(0), input_dim))
        outputs_fake = discriminator(fake_data)
        loss_g = -torch.mean(outputs_fake)
        loss_g.backward()

        optimizer_g.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Wasserstein distance: {wasserstein_distance:.4f}, Loss G: {loss_g:.4f}')
    wasserstein_distances.append(wasserstein_distance.detach())
    generator_losses.append(loss_g.detach())

# plot training process
plt.plot(list(range(1, num_epochs+1)), wasserstein_distances)
plt.xlabel('epoch')
plt.ylabel('approximate negative Wasserstein distance')
plt.show()

# test trained generator
test_data = generator(torch.randn(1000, input_dim)).detach()
test_df = pd.DataFrame(scaler.inverse_transform(test_data), columns=real_data_original.columns)

# empirical qq plots
for column in test_df.columns:
    plt.scatter(np.sort(real_data_original[column]), np.sort(test_df[column]))
    plt.axline((0, 0), slope=1)
    plt.xlabel('real data')
    plt.ylabel('generated data')
    plt.title(column)
    plt.show()