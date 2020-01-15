import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer
from utils.transformer import Embedder, PositionalEncoder, ClassificationHead, TransformerBlock 


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)



## TransformerClassification
class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256,
                           output_dim=2):
        super().__init__()
        
        # モデルの構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model, max_seq_len)
        self.net3_1 = TransformerBlock(d_model)
        self.net3_2 = TransformerBlock(d_model)
        self.net4 = ClassificationHead(d_model, output_dim)
        
    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1, normalized_weights_1 = self.net3_1(x2, mask)
        x3_2, normalized_weights_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)
        return x4, normalized_weights_1, normalized_weights_2

# パラメータの初期化を定義
def weights_init(m):
    classname =  m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)