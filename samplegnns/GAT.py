import torch
import torch.nn as nn

from torch_geometric.nn import GATConv

class GAT(nn.Module):
    """
    x --> GATConv --> BN --> ReLU --> Dropout --> x .....x --> GATConv --> out (logits)

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=1, dropout=.1):
        super().__init__()

        self.num_layers = num_layers

        # Convolution List
        conv_list = []
        conv_list.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers-2):
             conv_list.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads,  dropout=dropout))
        conv_list.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.convs = nn.ModuleList(conv_list)

        # Batch Norm List
        bn_list = []
        for _ in range(num_layers -1):
            bn_list.append(nn.BatchNorm1d(hidden_channels * heads))
        self.bns = nn.ModuleList(bn_list) 
        self.dropout = dropout


    def forward(self, x, edge_index):
        for i in range(self.num_layers -1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            # dropout is disabled during evaluation
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[self.num_layers-1](x, edge_index)

        out = x
        return out
