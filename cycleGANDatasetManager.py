import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import torchvision.utils as vutils
from config import zebra_training_data_path, zebra_testing_data_path, horse_training_data_path, horse_testing_data_path, image_size, batch_size, workers, device

class CycleGANDatasetManager:
    def __init__(self):
        self.horse_training_data = self.data2dataloader(horse_training_data_path)
        self.horse_testing_data = self.data2dataloader(horse_testing_data_path)
        self.zebra_training_data = self.data2dataloader(zebra_training_data_path)
        self.zebra_testing_data = self.data2dataloader(zebra_testing_data_path)
    
    def data2dataloader(self, dataset_path):
        dataset = datasets.ImageFolder(root=dataset_path,
                        transform=transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0, 0, 0), (1, 1, 1)),
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        return dataloader

    def display_dataset_sample(self, dataset, save_name):
        real_batch = next(iter(dataset))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig('./dataset_sample/'+save_name)
    
    def display_all_dataset_sample(self):
        self.display_dataset_sample(self.horse_training_data, 'horse_training_data.png')
        self.display_dataset_sample(self.horse_testing_data, 'horse_testing_data.png')
        self.display_dataset_sample(self.zebra_training_data, 'zebra_training_data.png')
        self.display_dataset_sample(self.zebra_testing_data, 'zebra_testing_data.png')
        
        
        
if __name__ == '__main__':
    #print('Hello World!')
    dataset_manager = CycleGANDatasetManager()
    #dataset_manager.display_all_dataset_sample()
    print('CycleGANDatasetManager initialized successfully.')