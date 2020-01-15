import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Transformer
## Embbeder
class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(
        embeddings=text_embedding_vectors, freeze=True)
        
    def forward(self, x):
        y = self.embeddings(x)
        return y

## Postional Encoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super(PositionalEncoder, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 単語ベクトルの次元数
        self.d_model = d_model
        
        # 単語の順番posとベクトルの次元位置iの(p, i)によって一意に定まる表を作成する
        pe = torch.zeros(max_seq_len, d_model)
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(
                                        pos / (10000 ** ((2*i)/d_model)))
                pe[pos, i+1] = math.cos(
                                        pos / (10000 ** ((2*(i+1))/d_model)))
        
        self.pe = pe.to(device).unsqueeze(0)
        
        # 勾配を計算しないようにする
        self.pe.requires_grad = False
        
    def forward(self, x):
        # 入力xとPositional Encoderを足し算する
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret

# Attentionの作成
class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        
        # 特徴量の作成
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        # 出力の全結合層
        self.out = nn.Linear(d_model, d_model)
        
        # Attentionの大きさ調整の変数
        self.d_k = d_model
        
    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Attentionの値を計算する
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        
        # maskを計算
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask==0, -1e9)
        
        # softmaxで規格化する
        normalized_weights = F.softmax(weights, dim=-1)
        
        # AttentionをValueと掛け算
        output = torch.matmul(normalized_weights, v)
        
        # 特徴量を変換
        output = self.out(output)
        
        return output, normalized_weights

## FeedForwardの作成
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

## TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # LayerNorm層
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        # Attention層
        self.attn = Attention(d_model)
        
        # 全結合層
        self.ff = FeedForward(d_model)
        
        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 正規化とAttention
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(
            x_normalized, x_normalized, x_normalized, mask)
        
        x2 = x + self.dropout_1(output)
        
        # 正規化と全結合層構築
        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))
        
        return output, normalized_weights

## ClassificationHead
class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()
        
        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)
        
        # 重み初期化
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def forward(self, x):
        x0 = x[:, 0, :]   # 各文の先頭の単語の特徴量を取り出す
        out = self.linear(x0)
        
        return out

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