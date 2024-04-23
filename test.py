import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms

from datasets import CONSEP
from resnet_unet import TwoEncodersOneDecoder
from utils.plotting import plot_loss

from train import eval_dice_with_h_x
cuda_device = torch.device('cuda', 0)
def test():

    dataset_test = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/test', mode='val')
    dataloader_test = DataLoader(dataset_test, batch_size=32, num_workers=10, pin_memory=True)

    model = TwoEncodersOneDecoder(resnet18, pretrained=False, out_channels=1)
    model.load_state_dict(torch.load('TwoEncodersOneDecoder_consep.pth')['model'])
    model.eval()
    cuda_device = torch.device('cuda', 0)
    model.to(cuda_device)  # Note: comment out this line if you'd like to run the model on the CPU locally


    dice_scores = eval_dice_with_h_x(model, dataloader_test)
    print(f'Dice Score for test set: {dice_scores}')


if __name__ == '__main__':
    test()
