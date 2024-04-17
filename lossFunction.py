import torch

def discriminatorLoss(real, fake):
    return (torch.mean((real - 1)**2) + torch.mean(fake**2))

def generatorLoss(fake):
    return torch.mean((fake - 1)**2)

cycleConsistencyLoss = torch.nn.L1Loss() 