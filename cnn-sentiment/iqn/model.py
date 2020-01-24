import torch
import torch.nn as nn
import torch.nn.functional as F

class IQN(nn.Module):
    def __init__(self, text_embedding_vector, vocab_size, embedding_dim,
                 n_filters, filter_sizes, pad_idx,
                 d_model=300, n_actions=1, n_quant=8):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_actions = n_actions
        self.n_quant = n_quant
        
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx)


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

    def forward(self, text, eta=1.0):
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
        tau = eta * torch.rand(mb_size, self.n_quant).to(self.device).view(mb_size, -1, 1)

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

        return action_value, tau

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.constant_(m.bias, 0)