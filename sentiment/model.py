import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer
from utils.transformer import Embedder, PositionalEncoder, ClassificationHead, TransformerBlock 


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



class CNN2(nn.Module):
    def __init__(self, text_embedding_vectors, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)


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


class QRDQN(nn.Module):
    def __init__(self, text_embedding_vector, vocab_size, embedding_dim, 
                    n_filters, filter_sizes, pad_idx,
                    d_model=300, num_actions=2, quantiles=51):
        super().__init__()

        self.num_actions = num_actions
        self.quantiles = quantiles

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 
                            self.num_actions * self.quantiles)

    def forward(self, text):
        embedded = self.embedding(text)    # [batch size, sen len, emb dim]

        embedded = embedded.unsqueeze(1)   # [batch size, 1, sen len, emb dim]

        h = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]   # [batch size, n_filters, sen len - filter_sizes[n] + 1]

        h = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in h]

        h = torch.cat(h, dim=1)

        h = self.fc(h)

        return h.view(-1, self.num_actions, self.quantiles)

class IQN(nn.Module):
    def __init__(self, text_embedding_vector, vocab_size, embedding_dim,
                 n_filters, filter_sizes, pad_idx,
                 d_model=300, n_actions=2, n_quant=8):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_actions = n_actions
        self.n_quant = n_quant

        self.phi = nn.Linear(64, 64)
        # self.phi_bias = nn.Parameter(torch.zeros(64))

        # self.fc0 = nn.Linear(n_state, 64)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc0 = nn.Linear(len(filter_sizes) * n_filters, 64)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, n_actions)

    def forward(self, text):
        # [batch_size, num_state]
        mb_size = text.size(0)

        # [batch_size, sen_len] → [batch_size, sen_len, emb_dim]
        embedded = self.embedding(text)
        
        # [batch size, 1, sen len, emb dim]
        embedded = embedded.unsqueeze(1)

        # [batch size, n_filters, sen len - filter_sizes[n] + 1]
        x = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # [batch size, n_filters, sen len - filter_sizes[n] + 1]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]

        # [batch size, n_filters * len(filter_sizes)]
        x = torch.cat(x, dim=1)

        # [batch size, 64]
        x = F.relu(self.fc0(x))

        # [batch_size, 64] 
        # → [batch_size, quant, 64]
        x = x.repeat(1, 1, self.n_quant).view(mb_size, self.n_quant, 64)

        # [batch_size, quant, 1]
        tau = torch.rand(mb_size, self.n_quant).to(self.device).view(mb_size, -1, 1)

        # [1, 64]
        pi_mtx = math.pi * torch.arange(0, 64, 1.0).to(self.device).unsqueeze(0)

        # [batch_size, quant, 64]
        cos_tau = torch.cos(torch.matmul(tau, pi_mtx))

        # [batch_size, quant, quantile_embedding_di64]
        phi = F.relu(self.phi(cos_tau))

        # [batch_size, quant, 4]
        h = x * phi

        # [batch_size, quant, 64]
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # [batch_size, quant, 2]
        action_value = self.fc_q(h).view(mb_size, self.n_quant, self.n_actions)
        # action_value = action_value.transpose(0, 2, 1)

        return action_value, tau

# パラメータの初期化を定義
def weights_init(m):
    classname =  m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)