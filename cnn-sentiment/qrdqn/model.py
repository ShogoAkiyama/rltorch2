import torch
import torch.nn as nn
import torch.nn.functional as F


class QRDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                    n_filters, filter_sizes, pad_idx,
                    num_actions=1, quantiles=51):
        super().__init__()

        self.num_actions = num_actions
        self.quantiles = quantiles

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ]).apply(weights_init_xavier)

        self.fc = nn.Linear(
            len(filter_sizes) * n_filters, 100).apply(weights_init_he)
        
        self.out_fc = nn.Linear(
            100, self.num_actions * self.quantiles).apply(weights_init_xavier)

    def forward(self, text):
        embedded = self.embedding(text)    # [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)   # [batch size, 1, sent len, emb dim]

        h = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]   # [batch size, n_filters, sent len - filter_sizes[n] + 1]

        h = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in h]

        h = torch.cat(h, dim=1)

        h = F.relu(self.fc(h))
        h = self.out_fc(h)

        return h.view(-1, self.num_actions, self.quantiles)


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.constant_(m.bias, 0)