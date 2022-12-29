import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_transformer_net1 import GraphTransformer
from models.transformer_net import Encoder, Decoder

if torch.cuda.is_available():
    device = torch.device('cuda')


class GTDTInet(nn.Module):
    def __init__(self, max_length=1000, compound_graph_dim=128, protein_dim=128, out_dim=2):
        super(GTDTInet, self).__init__()
        self.max_length = max_length
        self.compound_graph_dim = compound_graph_dim   #药物分子图
        self.protein_dim = protein_dim  #蛋白质

        self.compound_encoder = GraphTransformer(device=device, n_layers=10, node_dim=44, edge_dim=10, hidden_dim=128,
                                                 out_dim=128, n_heads=8, in_feat_dropout=0.0, dropout=0.2,
                                                 pos_enc_dim=8)   #药物分子encoder层
        self.protein_encoder = Encoder(n_layers=10, in_dim=100, embed_size=128, heads=4, forward_expansion=4,
                                       dropout=0.2)            #蛋白质encoder层
        self.fc1 = nn.Linear(self.max_length * 100, protein_dim)   #线性层

        self.classifier = nn.Sequential(
            nn.Linear(self.compound_graph_dim + self.protein_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, compound_graph, protein):
        compound_graph = self.compound_encoder(compound_graph)
        compound = compound_graph
        protein = self.protein_encoder(protein)
        protein = protein.view(-1, self.max_length * 100)
        protein = self.fc1(protein)
        x = torch.cat((compound, protein), dim=1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    teaher_model = GTDTInet(max_length=1000, compound_graph_dim=128, protein_dim=128, out_dim=2)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_num, trainable_num

    total_num, trainable_num = get_parameter_number(teaher_model)
    print('trainable_num:', trainable_num, 'total_num:', total_num)
    print("trainable_num, total_num: %.2f M, %.2f M" % (trainable_num / 1e6, total_num / 1e6))
