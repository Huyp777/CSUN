import torch
from torch import nn

class SentenceModule(nn.Module):
    def __init__(self):
        super(SentenceModule, self).__init__()
        self.lstm = nn.LSTM(
            300, 1024, num_layers=3,
            bidirectional=False, batch_first=True
        )
        self.fc = nn.Linear(1024, 256)
        self.net = nn.Sequential(
            nn.RReLU(),
            nn.Linear(256, 256)
        )
    def encode_query(self, queries, wordlens):
        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]
        queries = queries[range(queries.size(0)), wordlens.long() - 1]
        return self.fc(queries)

    def forward(self, queries, wordlens):
        feed = self.encode_query(queries, wordlens)[:,:]
        out = self.net(feed)
        return out

def build_Sentence():
    return SentenceModule()
