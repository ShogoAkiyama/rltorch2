import math 

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc0 = nn.Linear(len(filter_sizes) * n_filters, 64)

    def forward(self, embedded):
        # [batch size, 1, sen len, emb dim]
        embedded = embedded.unsqueeze(1)

        # [batch size, n_filters, sen len - filter_sizes[n] + 1]
        x = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # [batch size, n_filters, sen len - filter_sizes[n] + 1]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]

        # [batch size, n_filters * len(filter_sizes)]
        x = torch.cat(x, dim=1)

        return self.fc0(x)


class RNN(nn.Module):
    def __init__(self, emb_dim, h_dim, batch_first=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_dim = h_dim
        self.lstm = nn.LSTM(emb_dim, h_dim,
                            batch_first=batch_first, bidirectional=True)
        self.attn = Attn(h_dim)
        self.fc0 = nn.Linear(h_dim, 64)

    def init_hidden(self, b_size):
        h0 = torch.zeros(2, b_size, self.h_dim).to(self.device)
        c0 = torch.zeros(2, b_size, self.h_dim).to(self.device)
        return (h0, c0)

    def forward(self, emb, lengths=None):
        self.hidden = self.init_hidden(emb.size(0))
        # emb = self.embed(sentence)
        packed_emb = emb

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        # print(packed_emb.shape, ' ', self.hidden[0].shape, ' ', self.hidden[1].shape)
        out, hidden = self.lstm(packed_emb, self.hidden)

        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(out)[0]

        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        attns = self.attn(out)
        feats = (out * attns).sum(dim=1) # (b, s, h) -> (b, h)

        return self.fc0(feats), attns


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(24,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.fc(encoder_outputs.reshape(-1, self.h_dim))
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2)


class IQN(nn.Module):
    def __init__(self, text_vectors, vocab_size, embedding_dim,
                 n_filters, filter_sizes, pad_idx,
                 d_model=300, n_actions=1, n_quant=8, rnn=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn = rnn

        self.n_actions = n_actions
        self.n_quant = n_quant
        self.i = torch.arange(0, 64, 1.0).to(self.device).unsqueeze(0)

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=text_vectors, freeze=True)
        # self.embedding = nn.Embedding(
        #     vocab_size, 
        #     embedding_dim, 
        #     padding_idx=pad_idx)

        self.phi = nn.Linear(64, 64)
        # self.phi_bias = nn.Parameter(torch.zeros(64))

        self.cnn = CNN(embedding_dim, n_filters, filter_sizes)
        self.rnn = RNN(embedding_dim, n_filters)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, n_actions)

    def forward(self, text, eta=1.0, eval=False):
        # [batch_size, num_state]
        mb_size = text.size(0)

        # [batch_size, sen_len] → [batch_size, sen_len, emb_dim]
        embedded = self.embedding(text)

        if self.rnn:
            x, attn = self.rnn(embedded)
        else:
            x = self.cnn(embedded)

        # [batch size, 64]
        x = F.relu(x)

        # [batch_size, 64] → [batch_size, quant, 64]
        x = x.repeat(1, 1, self.n_quant).view(mb_size, self.n_quant, 64)

        # [batch_size, quant, 1]
        if eval:
            tau = eta * torch.linspace(0, 1, self.n_quant).to(self.device).unsqueeze(1)
            print(tau.shape)
            tau = tau.repeat(mb_size, 1).view(mb_size, -1, 1)
            print(tau.shape)
        else:
            tau = eta * torch.rand(mb_size, self.n_quant).to(self.device).view(mb_size, -1, 1)

        # [1, 64]
        pi_mtx = math.pi * self.i

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

        if self.rnn:
            return action_value, tau, attn
        else:
            return action_value, tau


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.constant_(m.bias, 0)
