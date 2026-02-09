import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(n)])
        self.norm = nn.LayerNorm(layer.size)