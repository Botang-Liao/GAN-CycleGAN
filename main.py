from model import CycleGAN
from cycleGANDatasetManager import CycleGANDatasetManager
from train import Trainer
from reference import Reference
from imageManager import TrainingManager

if __name__ == '__main__':
    
    data_loader = CycleGANDatasetManager()
    
    # training
    model = CycleGAN()
    model.load_models()
    training_manager = TrainingManager()
    trainer = Trainer(model, data_loader, training_manager)
    trainer.train()
    
    # reference
    model = CycleGAN()
    model.load_models()
    reference = Reference(model, data_loader)
    reference.plot_all_images()
    