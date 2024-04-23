from pathlib import Path

import torch
from torchvision.models import resnet18
from torchvision.utils import save_image

from resnet_unet import TwoEncodersOneDecoder


def visualise_segmentation(model_path, destination_path, dataloader):
    """
    Visualises the output of a model as an image with 3 columns:
    1st column is the original input image to be segmented.
    2nd column is the output of the model -> the segmentation mask.
    3rd column is the ground truth segmentation mask.

    :param model_path: The path to the model checkpoint file
    :param destination_path: The path to the directory where you'd like to save the images
    :param dataloader: A pytorch dataloader that provides the input images to be segmented
    """
    model = TwoEncodersOneDecoder(resnet18, pretrained=False, out_channels=1)
    #model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    cuda_device = torch.device('cuda', 0)
    model.to(cuda_device)  # Note: comment out this line if you'd like to run the model on the CPU locally

    destination_path = Path(destination_path)
    destination_path.mkdir(exist_ok=True)
    for batch_idx, (x, h_x, y) in enumerate(dataloader):
        # TODO: Move (x, h_x, y) to cuda if you're using GPU.
        x=x.to(cuda_device)
        h_x=h_x.to(cuda_device)
        y=y.to(cuda_device)

        # TODO: Convert h_x to have 3 channels just like you did in train.py
        h_x=h_x.expand(-1,3,-1,-1)
        

        with torch.no_grad():
            # TODO: Step 1) Convert h_x to have 3 channels just like you did in the train() function

            # TODO: Step 2) Run the model and store outputs in the variable out below
            out = model(x,h_x)

            # TODO: Step 3) Convert the outputs to a binary mask as follows:
            #               a) Pass the output through the sigmoid function to get an output between 0 and 1
            #               b) Using 0.5 as the threshold, convert the values to 0 if they are < 0.5 and 1 if > 0.5
            sigmoid = torch.sigmoid(out)
            out = (sigmoid > 0.5)

        edges = []
        # TODO: Convert each image in the batch "out" to have 3 channels and store them in the list "edges"
        out=out.expand(-1,3,-1,-1)

        for i in out:
            edges.append(i)
        

        assert len(edges) == x.size(0), f'Expected {x.size(0)} elements in edges but got {len(edges)} instead.'

        for i in range(len(edges)):
            img_name = str(batch_idx * dataloader.batch_size + i)
            # TODO: Convert y[i] to have 3 channels and store it in this variable "gt"

            y_channel = y[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            gt = y_channel.expand(1, 3, y_channel.shape[2], y_channel.shape[3])
            gt = gt.squeeze(0)


            image = torch.stack([x[i], edges[i], gt])
            save_image(image.float().to('cpu'),
                       destination_path / f'{img_name}.jpg',
                       padding=10, pad_value=0.5)


