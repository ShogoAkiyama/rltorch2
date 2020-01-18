import torch
import torch.nn as nn
import torch.nn.functional as F


class QRDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                    n_filters, filter_sizes, pad_idx,
                    num_actions=2, quantiles=51):
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
        embedded = self.embedding(text)    # [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)   # [batch size, 1, sent len, emb dim]

        h = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]   # [batch size, n_filters, sent len - filter_sizes[n] + 1]

        h = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in h]

        h = torch.cat(h, dim=1)

        h = self.fc(h)

        return h.view(-1, self.num_actions, self.quantiles)

