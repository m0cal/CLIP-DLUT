import torch 
import torch.nn as nn
from torch import Tensor

class GradientPredictor(nn.Module):
    def __init__(self, frozen=False, dim=512):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 3)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)
        
        if frozen == True:
            for p in self.mlp.parameters():
                p.requires_grad = False
        
    def freeze(self):
        for p in self.mlp.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.mlp.parameters():
            p.requires_grad = True
    
    def forward(self, color: Tensor):
        """
        :param color: a 3D Tensor, [R, G, B] value in [0, 1]
        :type color: Tensor
        """

        delta = self.mlp(color)

        return delta
