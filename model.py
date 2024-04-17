import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device, norm_layer, models_name, lr



class CycleGAN():
    def __init__(self, f=64, blocks=9):
        self.zebra2HorseGenerator = Generator(f, blocks).to(device)
        self.horse2ZebraGenerator = Generator(f, blocks).to(device)
        self.horseDiscriminator = Discriminator().to(device)
        self.zebraDiscriminator = Discriminator().to(device)
        
        self.optimizer_zebra2HorseGenerator = torch.optim.Adam(self.zebra2HorseGenerator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_horse2ZebraGenerator = torch.optim.Adam(self.horse2ZebraGenerator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_horseDiscriminator = torch.optim.Adam(self.horseDiscriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_zebraDiscriminator = torch.optim.Adam(self.zebraDiscriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        
        
    def save_models(self):
        torch.save(self.zebra2HorseGenerator, "./model/"+models_name+"_Zebra2HorseGenerator.pth")
        torch.save(self.horse2ZebraGenerator, "./model/"+models_name+"_Horse2ZebraGenerator.pth")
        torch.save(self.horseDiscriminator, "./model/"+models_name+"_HorseDiscriminator.pth")
        torch.save(self.zebraDiscriminator, "./model/"+models_name+"_ZebraDiscriminator.pth")

    def load_models(self):
        self.zebra2HorseGenerator=torch.load("./model/"+models_name+"_Zebra2HorseGenerator.pth", map_location=device)
        self.horse2ZebraGenerator=torch.load("./model/"+models_name+"_Horse2ZebraGenerator.pth", map_location=device)
        self.horseDiscriminator=torch.load("./model/"+models_name+"_HorseDiscriminator.pth", map_location=device)
        self.zebraDiscriminator=torch.load("./model/"+models_name+"_ZebraDiscriminator.pth", map_location=device)

        
class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = norm_layer(f)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))

class Generator(nn.Module):
    def __init__(self, f=64, blocks=9):
        super(Generator, self).__init__()
        
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f), nn.ReLU(True),
                  nn.Conv2d(  f, 2*f, 3, 2, 1), norm_layer(2*f), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), norm_layer(4*f), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
        layers.extend([
                nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), 
                                            norm_layer(2*f), nn.ReLU(True),
                nn.ConvTranspose2d(2*f,   4*f, 3, 1, 1), nn.PixelShuffle(2), 
                                            norm_layer(  f), nn.ReLU(True),
                nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh()])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):  
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc,ndf,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf,ndf*2,4,2,1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4,ndf*8,4,1,1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15
            nn.Conv2d(ndf*8,1,4,1,1)
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        return self.main(input)    
    
    
if __name__ == '__main__':

    model = CycleGAN()
    print(model.zebra2HorseGenerator)