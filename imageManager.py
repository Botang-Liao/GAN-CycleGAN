import torch
from collections import deque
import random
from config import DEQUE_SIZE, RANDOM_SAMPLE_NUMBER

class TrainingManager:
    def __init__(self):
        self.fake_horse = deque(maxlen=DEQUE_SIZE)
        self.fake_zebra = deque(maxlen=DEQUE_SIZE)

    def updateOldFakeHorse(self, fake_horse):
        sub_tensors = fake_horse.split(1, dim=0)
        for sub_tensor in sub_tensors:
            self.fake_horse.append(sub_tensor)
       
    def updateOldFakeZebra(self, fake_zebra):
        sub_tensors = fake_zebra.split(1, dim=0)
        for sub_tensor in sub_tensors:
            self.fake_zebra.append(sub_tensor)
        
    # def getRamdonFakeHorseSample(self):
    #     if RANDOM_SAMPLE_NUMBER > len(self.fake_horse):
    #         chosen_horse = torch.stack(list(self.fake_horse))
    #         concatenated_tensor = torch.cat(chosen_horse, dim=0)
    #         return concatenated_tensor
    #     else:
    #         chosen_horse = random.sample(self.fake_horse, RANDOM_SAMPLE_NUMBER)
    #         concatenated_tensor = torch.cat(chosen_horse, dim=0)
    #         return concatenated_tensor
    def getRamdonFakeHorseSample(self):
        return torch.stack(list(self.fake_horse)).view(-1, 3, 256, 256)

    def getRandomFakeZebraSample(self):
        return torch.stack(list(self.fake_zebra)).view(-1, 3, 256, 256)
        