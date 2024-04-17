from model import CycleGAN
from cycleGANDatasetManager import CycleGANDatasetManager
from config import epochs, device, generator_loss_weight, cycle_consistency_loss_weight, identity_loss_weight
from lossFunction import discriminatorLoss, generatorLoss, cycleConsistencyLoss
from reference import Reference
from imageManager import TrainingManager
class Trainer:
    def __init__(self, model:CycleGAN, data_loader:CycleGANDatasetManager, training_manager:TrainingManager):
        self.model = model
        self.data_loader = data_loader
        self.horse_training_data = data_loader.horse_training_data
        self.zebra_training_data = data_loader.zebra_training_data
        self.horse_testing_data = data_loader.horse_testing_data
        self.zebra_testing_data = data_loader.zebra_testing_data
        self.training_manager = training_manager
        self.horse_discriminate_loss = []
        self.zebra_discriminate_loss = []
        self.generator_loss = []
        
    
    def train(self):
        for epoch in range(epochs):
            self.model.zebra2HorseGenerator.train()
            self.model.horse2ZebraGenerator.train()
            for (data_horse, data_zebra) in zip(self.horse_training_data, self.zebra_training_data):
    
                real_horse = data_horse[0].to(device)
                real_zebra = data_zebra[0].to(device)
                
                # Generated images
                fake_horse = self.model.zebra2HorseGenerator(real_zebra)
                reconstructed_zebra = self.model.horse2ZebraGenerator(fake_horse)
                identity_zebra = self.model.horse2ZebraGenerator(real_zebra)
                self.training_manager.updateOldFakeHorse(fake_horse.detach())
                
                fake_zebra = self.model.horse2ZebraGenerator(real_horse)
                reconstructed_horse = self.model.zebra2HorseGenerator(fake_zebra)
                identity_horse = self.model.zebra2HorseGenerator(real_horse)
                self.training_manager.updateOldFakeZebra(fake_zebra.detach())
                
                self.trainHorseDiscriminator(real_horse, self.training_manager.getRamdonFakeHorseSample())
                self.trainZebraDiscriminator(real_zebra, self.training_manager.getRandomFakeZebraSample())
                self.trainGenerator(real_horse, real_zebra, fake_horse, fake_zebra, reconstructed_horse, reconstructed_zebra, identity_horse, identity_zebra)
               
            print(f'Epoch {epoch+1}/{epochs} Discriminator Loss: Horse {self.horse_discriminate_loss[-1]} Zebra {self.zebra_discriminate_loss[-1]} Generator Loss: {self.generator_loss[-1]}')
            self.model.save_models()  
            reference = Reference(self.model, self.data_loader)
            reference.plot_all_images() 
    
    def trainHorseDiscriminator(self, real_horse, fake_horse):
        self.model.optimizer_horseDiscriminator.zero_grad()
        horse_discriminator_loss = discriminatorLoss(self.model.horseDiscriminator(real_horse), self.model.horseDiscriminator(fake_horse))
        self.horse_discriminate_loss.append(horse_discriminator_loss.item())
        horse_discriminator_loss.backward()
        self.model.optimizer_horseDiscriminator.step()      
        
    def trainZebraDiscriminator(self, real_zebra, fake_zebra):
        self.model.optimizer_zebraDiscriminator.zero_grad()
        zebra_discriminator_loss = discriminatorLoss(self.model.zebraDiscriminator(real_zebra), self.model.zebraDiscriminator(fake_zebra))
        self.zebra_discriminate_loss.append(zebra_discriminator_loss.item())
        zebra_discriminator_loss.backward()
        self.model.optimizer_zebraDiscriminator.step()        
        
    def trainGenerator(self,real_horse, real_zebra, fake_horse, fake_zebra, reconstructed_horse, reconstructed_zebra, identity_horse, identity_zebra):
        self.model.optimizer_zebra2HorseGenerator.zero_grad()
        self.model.optimizer_horse2ZebraGenerator.zero_grad()
        generator_loss = generatorLoss(self.model.zebraDiscriminator(fake_zebra)) * generator_loss_weight
        generator_loss += generatorLoss(self.model.horseDiscriminator(fake_horse)) * generator_loss_weight
        generator_loss += cycleConsistencyLoss(real_horse, reconstructed_horse) * cycle_consistency_loss_weight
        generator_loss += cycleConsistencyLoss(real_zebra, reconstructed_zebra) * cycle_consistency_loss_weight
        generator_loss += cycleConsistencyLoss(real_zebra, identity_zebra) * identity_loss_weight
        generator_loss += cycleConsistencyLoss(real_horse, identity_horse) * identity_loss_weight
        self.generator_loss.append( generator_loss.item())
        generator_loss.backward()
        self.model.optimizer_zebra2HorseGenerator.step()
        self.model.optimizer_horse2ZebraGenerator.step()