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

cuda_device = torch.device('cuda', 0)


def save_checkpoint(model, name):
    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, f'{name}.pth')


def dice_loss_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Soft dice loss = 2*|A∩B| / |A|+|B|
    Note: x and target tensors should have values between 0 and 1
    """
    eps = 1e-7
    numerator = 2 * (x * target).sum((1, 2))
    denominator = (x + target).sum((1, 2))

    dice = 1 - (numerator + eps) / (denominator + eps)
    return dice


def train():
    model = TwoEncodersOneDecoder(resnet18, pretrained=True, out_channels=1)
    model.train()
    model.to(cuda_device)

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomPosterize(3),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.3, 0.3]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.RandomPosterize(3),
        transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  # New: Randomly flip the image vertically
        transforms.RandomRotation(degrees=30),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.2),  # New: Random affine transformation
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # New: Random Gaussian blur
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    dataset = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/train', mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)

    dataset_val = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/val', mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=32, num_workers=10, pin_memory=True)

    num_epochs = 30

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dataloader) * num_epochs))

    best_dice_score = 0
    best_epoch = 0
    bce_losses = []
    dice_losses = []
    for epoch in range(0, num_epochs):
        print(f'Epoch: {epoch}')
        epoch_start_time = time.time()
        for batch_idx, (x, h_x, y) in enumerate(dataloader, 1):
            # TODO: Step 1) Move x, h_x, and y to GPU
            x=x.to(cuda_device)
            h_x=h_x.to(cuda_device)
            y=y.to(cuda_device)

            # TODO: Step 2) Convert h_x to have 3 channels by repeating the 1 channel it has 3 times.
            #               Hint: You can use h_x.expand() function to do that without increasing memory usage
            #                     or use the .repeat() function
            h_x = h_x.expand(-1, 3, -1, -1)

            # TODO: Step 3) Run the model and get the outputs.
            out = model(x, h_x)

            # TODO: Step 4) a) Call the loss functions bce_loss_fn & dice_loss_fn. Add them to get the loss.
            #                  The loss should be a single number (not an array).
            #                   Hint: Use .mean() on dice_loss_fn's output
            #               b) Append the loss values to their respective lists for plotting.
            #                  Use .item() while appending the values.
            #print(f'out: {out.size()}')
            #print(f'Y: {y.size()}')
            y = y.unsqueeze(1)
            #print(f'Y unsqueezed: {y.size()}')

            bce_loss = bce_loss_fn(out, y)
            dice_loss = dice_loss_fn(torch.sigmoid(out),y)

            loss = bce_loss + dice_loss.mean()

            bce_losses.append(bce_loss.item())
            dice_losses.append(dice_loss.mean().item())

            # TODO: Step 5) Run the backward() pass on the loss function
            loss.backward()

            # TODO: Step 6) Call the optimizer to update the model and then zero out the gradients.
            optimizer.step()
            optimizer.zero_grad()

            # The lines below prints loss values every 5 batches.
            # Uncomment them to see the loss go down during training.

            if batch_idx % 5 == 0 or batch_idx == len(dataloader) - 1:
                print(f'{epoch}-{batch_idx:03}\t{round(bce_loss.item(), 6)} {round(dice_loss.mean().item(), 6)} ', flush=True)

        scheduler.step()
        print(f'Epoch {epoch} took {timedelta(seconds=time.time() - epoch_start_time)}', flush=True)

        print('EVALUATING dice score on validation set')
        eval_start_time = time.time()
        dice_score_val = eval_dice_with_h_x(model, dataloader_val)
        print(f'Evaluation after epoch {epoch} took {timedelta(seconds=time.time() - eval_start_time)}', flush=True)
        if dice_score_val > best_dice_score:
            best_epoch = epoch
            best_dice_score = dice_score_val
            print('Saving model as a new best score has been achieved.')
            save_checkpoint(model, f'TwoEncodersOneDecoder_consep')

    print(f'Best dice score achieved on validation dataset was {best_dice_score} for epoch {best_epoch}', flush=True)
    # Save loss values in case the plotting throws an error or you wanna plot with different parameters
    with open('bce_loss.npy', 'wb') as f:
        np.save(f, np.array(bce_losses))
    with open('dice_loss.npy', 'wb') as f:
        np.save(f, np.array(dice_losses))
    # Save loss plots
    print('Saving plots')
    plot_loss(bce_losses, 'bce_loss')
    plot_loss(dice_losses, 'dice_loss')
    print('Plots saved')


def eval_dice_with_h_x(model, dataloader):
    model.eval()
    dice = []
    for batch_idx, (x, h_x, y) in enumerate(dataloader):
        # TODO: Move (x, h_x, y) to cuda
        x=x.to(cuda_device)
        h_x=h_x.to(cuda_device)
        y=y.to(cuda_device)

        with torch.no_grad():
            # TODO: Step 1) Convert h_x to have 3 channels just like you did in the train() function
            h_x = h_x.expand(-1, 3, -1, -1)

            # TODO: Step 2) Run the model and store outputs in the variable out below
            out = model(x, h_x)

            # TODO: Step 3) Convert the outputs to a binary mask as follows:
            #               a) Pass the output through the sigmoid function to get an output between 0 and 1
            #               b) Using 0.5 as the threshold, convert the values to 0 if they are < 0.5 and 1 if > 0.5
            sig = torch.sigmoid(out)
            mask = (sig > 0.5).float()
            dice_loss = dice_loss_fn(mask, y)
       
        dice.append(dice_loss)  # TODO: Replace None with the output of the dice_loss_fn called for the binary mask
    print(torch.cat(dice,0).mean())
    dice_score = 1 - torch.cat(dice, 0).mean()
    print(f'dice score (the higher the better): {dice_score}')
    model.train()
    return dice_score


if __name__ == '__main__':
    train()
