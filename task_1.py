from resnet_unet import Encoder, Decoder
from datasets import CONSEP
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomPosterize(3),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.3, 0.3])
    ])

dataset = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/train', mode='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)


encoder_1 = Encoder(resnet18)
encoder_1.train()


print("Testing Number of outputs for encoder")

for batch_idx, (x, h_x, y) in enumerate(dataloader, 1):
    blocks = encoder_1(x[0].unsqueeze(0))
    for indx, block in enumerate(blocks):
        print(f'Block {indx+1}: {block.shape}')
    break