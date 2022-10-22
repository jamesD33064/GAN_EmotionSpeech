import generator
import discriminator
import Layer

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
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

#load data
image_folder = ImageFolder(DataSet_path, transform=None, target_transform=None)
print(image_folder.class_to_idx)

# Model
latent_dim = 10
G = generator.Generator(latents=latent_dim).to(device)
D = discriminator.Discriminator().to(device)
G.apply(Layer.weights_init)
D.apply(Layer.weights_init)

#loss function
criterion = nn.CrossEntropyLoss()
#optimizer
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=5, gamma=0.5)
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.5)



for epoch in range(epochs):
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) 
        
        if CUDA:
            data, target = data.cuda(), target.cuda()

        # clear gradient
        optimizer.zero_grad()

        # Forward propagation
        output = model(data) 
        loss = criterion(output, target) 

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        predicted = torch.max(output.data, 1)[1]
        train_loss += loss.item()