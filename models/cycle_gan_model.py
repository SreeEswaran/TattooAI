import torch
import torch.nn as nn

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

    def generate(self, tattoo_tensor, body_tensor):
        return self.forward(tattoo_tensor)
