import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

class H2GCN(nn.Module):
    """
    x --> GCNConv --> BN --> ReLU --> Dropout --> x .....x --> GCNConv --> out (logits)

    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.1):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        #allow the input to be ego + 2 hidden outputs
        self.fc = nn.Linear(in_channels + hidden_channels + hidden_channels, out_channels)
        self.dropout = dropout


    def forward(self, x, edge_index):
        #Ego
        x0 = x

        # 1st Hop
        x1 = self.conv1(x0, edge_index)
        x1 = self.bn1(x1)
        x1 = nn.functional.relu(x1)
        x1 = nn.functional.dropout(x1, p=self.dropout, training=self.training)

        #2nd Hop
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = nn.functional.relu(x2)
        x2 = nn.functional.dropout(x2, p=self.dropout, training=self.training)

        out = self.fc(torch.cat([x0,x1,x2], dim=-1))

        return out
        

        
