import torch
import torch.nn as nn

from torch_geometric.nn import GCN2Conv

class GCN2(nn.Module):
    """
    https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GCN2Conv.html#torch_geometric.nn.conv.GCN2Conv
    x --> GCN2Conv --> BN --> ReLU --> Dropout --> x .....x --> GCN2Conv --> out (logits)

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=.1, alpha=0.1, theta=0.5):
        super().__init__()

        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_channels, hidden_channels)

        # Convolution List
        conv_list = []
        for i in range(num_layers):
            conv_list.append(GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=i + 1))
        
        self.convs = nn.ModuleList(conv_list)

        self.fc2 = nn.Linear(hidden_channels, out_channels)

        # Batch Norm List
        bn_list = []
        for _ in range(num_layers):
            bn_list.append(nn.BatchNorm1d(hidden_channels))
        self.bns = nn.ModuleList(bn_list)

        self.dropout = dropout


    def forward(self, x, edge_index):

        x = x_0 = nn.functional.relu(self.fc1(x))

        for i in range(self.num_layers):
            # dropout is disabled during evaluation
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, x_0, edge_index)
            x = self.bns[i](x)
            x = nn.functional.relu(x)

        x = self.fc2(x)   
        

        out = x
        return out
        

        
