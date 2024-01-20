import torch
import torch.nn as nn

from config import VisionProjectorConfig


class VisionProjector(nn.Module):

    '''
    input_dim := output dim of clip image emvbedding
    output_dim:= input dim for phi2 model

    '''
    def __init__(self, config: VisionProjectorConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_tokens = config.num_tokens

        self.pre_norm = nn.LayerNorm(self.input_dim)

        '''
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.num_tokens*self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.num_tokens*self.hidden_dim, self.num_tokens*self.output_dim)
        )
        '''
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.input_dim, self.num_tokens * self.output_dim)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.proj(x)
        x = x.reshape( (-1, self.num_tokens, self.output_dim) )
        return x