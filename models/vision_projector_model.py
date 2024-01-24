import torch
import torch.nn as nn

from config import VisionProjectorConfig

'''
class VisionProjector(nn.Module):

    def __init__(self, config: VisionProjectorConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_tokens = config.num_tokens

        self.pre_norm = nn.LayerNorm(self.input_dim)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.input_dim, self.num_tokens * self.output_dim)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.proj(x)
        x = x.reshape( (-1, self.num_tokens, self.output_dim) )
        return x

'''

class VisionProjector(nn.Module):

    def __init__(self, config: VisionProjectorConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.proj = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.proj(x)
        return x