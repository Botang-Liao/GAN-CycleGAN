import numpy as np
from config import device, reference_image_number
from model import CycleGAN
from cycleGANDatasetManager import CycleGANDatasetManager
import torchvision.utils as vutils
import matplotlib.pylab as plt


class Reference:
    def __init__(self, model=CycleGAN(), data_loader=CycleGANDatasetManager()):
        self.Horse2ZebraGenerator = model.horse2ZebraGenerator.eval()
        self.Zebra2HorseGenerator = model.zebra2HorseGenerator.eval()
        self.zebra = data_loader.zebra_testing_data
        self.horse = data_loader.horse_testing_data
        

    def plot_all_images(self):
        # 用 Gererator 把馬變成斑馬
        horse_data = next(iter(self.horse))[0].to(device)
        real_a_horse = horse_data.cpu().detach()
        horse2zebra = self.Horse2ZebraGenerator(horse_data)
        horse2zebra2horse = self.Zebra2HorseGenerator(horse2zebra).cpu().detach()
        horse2zebra = horse2zebra.cpu().detach()
        self.plot_comparison_diagram(real_a_horse, horse2zebra, horse2zebra2horse, 'Horse2Zebra')
        
        # 用 Generator 把斑馬變成馬
        zebra_data = next(iter(self.zebra))[0].to(device)
        real_a_zebra = zebra_data.cpu().detach()
        zebra2horse = self.Zebra2HorseGenerator(zebra_data)
        zebra2horse2zebra = self.Horse2ZebraGenerator(zebra2horse).cpu().detach()
        zebra2horse = zebra2horse.cpu().detach()
        self.plot_comparison_diagram(real_a_zebra, zebra2horse,  zebra2horse2zebra, 'Zebra2Horse')

        print('All images plotted successfully.')
        
    
    def plot_comparison_diagram(self, real_picture, fake_picture, reconstructed_picture, picture_name):
       
        fig, axs = plt.subplots(3, 1, figsize=(10, 6)) # 1 row, 3 columns
        
        # Plot real picture
        axs[0].imshow(np.transpose(vutils.make_grid((real_picture[:reference_image_number]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
        axs[0].axis("off")
        axs[0].set_title('Real Picture')

        # Plot fake picture
        axs[1].imshow(np.transpose(vutils.make_grid((fake_picture[:reference_image_number]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
        axs[1].axis("off")
        axs[1].set_title('Fake Picture')

        # Plot reconstructed picture
        axs[2].imshow(np.transpose(vutils.make_grid((reconstructed_picture[:reference_image_number]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
        axs[2].axis("off")
        axs[2].set_title('Reconstructed Picture')

        plt.tight_layout()
        plt.savefig('./result/'+picture_name) # Save the combined figure
        plt.close()
        
    


if __name__ == '__main__':
    reference = Reference()
    reference.plot_all_images()
    print('Reference initialized successfully.')