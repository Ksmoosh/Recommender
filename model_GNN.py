import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj
from IGMC.util_functions import *
from IGMC.data_utils import *
from IGMC.preprocessing import *


class IGMC(torch.nn.Module):
    def __init__(self, side_features=False, n_side_features=0):
        """
        Args:
            side_features: Zmienna boolowska okreslająca, 
                czy w modelu mają istniec dodatkowe atrybuty filmow
            n_sude_features: Liczba dodatkowych atrybutow filmow
        """
        super(IGMC, self).__init__()
        self.rel_graph_convs = torch.nn.ModuleList()
        self.rel_graph_convs.append(RGCNConv(in_channels=4, out_channels=32,\
                                             num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32\
                                             , num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32,\
                                             num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32,\
                                             num_relations=5, num_bases=4))
        if side_features:
            self.linear_layer1 = Linear(256 + n_side_features, 128)
        else:
            self.linear_layer1 = Linear(256, 128)
        self.linear_layer2 = Linear(128, 1)
        self.side_features = side_features

    def reset_parameters(self):
        """
        Reset parametrow warstw sieci
        """
        self.linear_layer1.reset_parameters()
        self.linear_layer2.reset_parameters()
        for i in self.rel_graph_convs:
            i.reset_parameters()

    def forward(self, data):
        """
        Glowna metoda sprzezenia w przod.
        Args:
            data: Dane na wejsciu sieci
        """
        num_nodes = len(data.x)
        edge_index_dr, edge_type_dr = dropout_adj(data.edge_index, data.edge_type,\
                                p=0.2, num_nodes=num_nodes, training=self.training)

        out = data.x
        h = []
        for conv in self.rel_graph_convs:
            out = conv(out, edge_index_dr, edge_type_dr)
            out = torch.tanh(out)
            h.append(out)
        h = torch.cat(h, 1)
        h = [h[data.x[:, 0] == True], h[data.x[:, 1] == True]]
        g = torch.cat(h, 1)
        if self.side_features:
            g = torch.cat([g, data.v_feature], 1)
        out = self.linear_layer1(g)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear_layer2(out)
        out = out[:,0]
        return out