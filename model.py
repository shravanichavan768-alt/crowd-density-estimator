import torch
import torch.nn as nn
import torchvision.models as models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # VGG16 first 10 conv layers
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend  = make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._load_vgg_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _load_vgg_weights(self):
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = list(vgg16.features.children())
        our_features = list(self.frontend.children())

        for i, (vgg_layer, our_layer) in enumerate(zip(vgg_features, our_features)):
            if isinstance(our_layer, nn.Conv2d) and isinstance(vgg_layer, nn.Conv2d):
                our_layer.weight.data = vgg_layer.weight.data
                our_layer.bias.data   = vgg_layer.bias.data


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)