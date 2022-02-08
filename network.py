import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, nx, nu, n_joints):
        super(Network, self).__init__()

        self.nx = nx
        self.nu = nu
        self.n_joints = n_joints

        self.layer1 = nn.Sequential(
            nn.Linear(
                in_features=nx,
                out_features=16,
            ),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(
                in_features=16,
                out_features=32,
            ),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=64,
            ),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(
                in_features=64,
                out_features=64,
            ),
            nn.ReLU(),
        )

        # # Three jointed pendulum:

        # self.layer5 = nn.Sequential(
        #     nn.Linear(
        #         in_features=64,
        #         out_features=128,
        #     ),
        #     nn.ReLU(),
        # )

        # Split into n_joints heads
        self.last_layers = nn.ModuleList()
        for _ in range(self.n_joints):
            self.last_layers.append(
                nn.Linear(
                    in_features=64,
                    out_features=nu,
                ),
            )

        self.init_weights()

    def init_weights(self):
        # Recommended weights initialization if using ReLU activation functions
        nn.init.kaiming_normal_(self.layer1[0].weight)
        nn.init.kaiming_normal_(self.layer2[0].weight)
        nn.init.kaiming_normal_(self.layer3[0].weight)
        nn.init.kaiming_normal_(self.layer4[0].weight)

        # # Three jointed pendulum:
        # nn.init.kaiming_normal_(self.layer5[0].weight)

        for _, layer in enumerate(self.last_layers):
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # # Three jointed pendulum:
        # x = self.layer5(x)

        # Get data from each (n_joints) head layer
        y = []
        for _, layer in enumerate(self.last_layers):
            y.append(layer(x))

        # Dimension is: n_joints, batch_size, nu
        
        # Reshape dimension to get: batch_size, j_joints, nu
        y = torch.stack(y, dim=1)

        return y
