import generator
import discriminator
import Layer

import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
# from torchvision.utils import save_image

DataSet_path=''

flag_gpu = 1
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 100
# Number of training epochs
epochs = 20
# Learning rate for optimizers
lr = 0.0002

# GPU
device = 'cuda:0' if (torch.cuda.is_available() & flag_gpu) else 'cpu'
print('GPU State:', device)
# Model
latent_dim = 10
G = generator.Generator(latents=latent_dim).to(device)
D = discriminator.Discriminator().to(device)
G.apply(Layer.weights_init)
D.apply(Layer.weights_init)

# Settings
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=5, gamma=0.5)
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.5)

# # Load data
# train_set = datasets.MNIST(DataSet_path, train=True, download=False, transform=transforms.ToTensor())
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
