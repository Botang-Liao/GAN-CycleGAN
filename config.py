import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

norm_layer = nn.InstanceNorm2d

zebra_training_data_path = '/HDD/n66104571/zebra_train'
zebra_testing_data_path = '/HDD/n66104571/zebra_test'
horse_training_data_path = '/HDD/n66104571/horses_train'
horse_testing_data_path = '/HDD/n66104571/horses_test'

batch_size = 16
workers = 2
image_size = (256,256)


# Learning rate for optimizers
lr = 0.0002

models_name = 'basic_training200'

epochs=50

reference_image_number = 5

generator_loss_weight = 1

cycle_consistency_loss_weight = 3

identity_loss_weight = 3

DEQUE_SIZE = 50

RANDOM_SAMPLE_NUMBER = 10