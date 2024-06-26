import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class Encoder(nn.Module):
    """Encoder with ResNet18 or ResNet34 encoder"""
    def __init__(self, encoder, *, pretrained=False):
        super().__init__()
        self.encoder = encoder(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        if not pretrained:
            self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return [block1, block2, block3, block4, block5]


class Decoder(nn.Module):
    """Decoder for two ResNet18 or ResNet34 encoders."""
    def __init__(self, out_channels=1):
        super().__init__()
        # TODO: Initialise the layers with correct number of input and output channels
        self.up_conv6 = up_conv(512 + 512,512)  # use the up_conv() function. Number of output channels = 512
        self.conv6 = double_conv(512 +256+256  , 512)  # use the double_conv() function. Number of output channels = 512
        self.up_conv7 = up_conv(512, 256)  # use the up_conv() function. Number of output channels = 256
        self.conv7 = double_conv(256 + 128 +128, 256)  # use the double_conv() function. Number of output channels = 256
        self.up_conv8 = up_conv(256,128)  # use the up_conv() function. Number of output channels = 128
        self.conv8 = double_conv(128 + 64+64, 128)  # use the double_conv() function. Number of output channels = 128
        self.up_conv9 = up_conv(128, 64)  # use the up_conv() function. Number of output channels = 64
        self.conv9 = double_conv(64+ 64 + 64, 64)  # use the double_conv() function. Number of output channels = 64
        self.up_conv10 = up_conv(64, 32)  # use the up_conv() function. Number of output channels = 32
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)  # Use nn.Conv2d with kernel size 1 to get the segmentation mask

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder1_blocks, encoder2_blocks):
        # TODO: Implement the forward which concatenates the outputs of the two encoders
        #       at each stage/block to use as the input
        # TODO: Replace the 1st "1" below in torch.Size with your batch size
 
        layer_1 = self.up_conv6(torch.cat((encoder1_blocks[4], encoder2_blocks[4]),dim=1))
        layer_2 = self.up_conv7(self.conv6(torch.cat((layer_1, encoder1_blocks[3], encoder2_blocks[3]),dim=1)))
        layer_3 = self.up_conv8(self.conv7(torch.cat((layer_2, encoder1_blocks[2], encoder2_blocks[2]),dim=1)))
        layer_4 = self.up_conv9 (self.conv8(torch.cat((layer_3, encoder1_blocks[1], encoder2_blocks[1]),dim=1)))
        layer_5 = self.up_conv10(self.conv9(torch.cat((layer_4, encoder1_blocks[0], encoder2_blocks[0]),dim=1)))
        output = self.conv10(layer_5)
        #print(f'Output SHAPE: {output.shape}')
        #assert output.shape == torch.Size([1, 1, 224, 224]), \
        #    f"The output shape should be same as the input image's shape but it is {output.shape} instead."
        return output


class TwoEncodersOneDecoder(nn.Module):
    def __init__(self, encoder, pretrained=True, out_channels=1):
        """
        The segmentation model to be used.
        :param encoder: resnet18 or resnet34 constructor to be used as the encoder
        :param pretrained: If True(default), the encoder will be initialised with weights
                           from the encoder trained on ImageNet
        :param out_channels: Number of output channels. The value should be 1 for binary segmentation.
        """
        super().__init__()
        # TODO: Initialise the encoders and the decoder
        self.encoder1 = Encoder(encoder=encoder, pretrained=pretrained)
        self.encoder2 = Encoder(encoder=encoder, pretrained=pretrained)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, x, h_x):    
        return self.decoder(self.encoder1(x), self.encoder2(h_x))
