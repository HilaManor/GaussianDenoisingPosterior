from torch import nn

class CNNDenoiser(nn.Module):
    def __init__(
            self,
            num_layers=10,
        ):

        super().__init__()

        layers = [
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
            ]
        layers += [
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
