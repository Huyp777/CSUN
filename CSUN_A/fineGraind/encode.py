import torch
from torch import nn
from torch.functional import F

feat_hidden_size = 512
query_input_size = 300
query_hidden_size = 512
num_layers = 3

class queryEncode(nn.Module):
    def __init__(self, feat_hidden_size, query_input_size, query_hidden_size, num_layers):
        super(queryEncode, self).__init__()
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers,
            bidirectional=False, batch_first=True
        )
        self.fc = nn.Linear(query_hidden_size, feat_hidden_size)
        self.conv = nn.Conv2d(feat_hidden_size, feat_hidden_size, 1, 1)

    def encode_query(self, queries, wordlens):
        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]
        queries = queries[range(queries.size(0)), wordlens.long() - 1]
        return self.fc(queries)

    def forward(self, queries, wordlens, map2d):
        queries = self.encode_query(queries, wordlens)[:,:,None,None]
        map2d = self.conv(map2d)
        return F.normalize(queries * map2d)

def build_encode():
    return queryEncode(feat_hidden_size, query_input_size, query_hidden_size, num_layers)
