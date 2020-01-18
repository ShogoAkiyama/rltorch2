import os

import torchtext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import tokenizer_with_preprocessing
from model import DQN

NEWS_PATH = os.path.join('..', 'data', 'news')

# Dataの取り出し
max_length = 256
batch_size = 32

# 読み込んだ内容に対して行う処理を定義
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                            init_token="<cls>", eos_token="<eos>")
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

train_ds = torchtext.data.TabularDataset.splits(
    path=NEWS_PATH, train='text_train.tsv',
    format='tsv',
    fields=[('Text1', TEXT), ('Text2', TEXT), ('Label', LABEL)])
train_ds = train_ds[0]

TEXT.build_vocab(train_ds,
                 min_freq=10)
TEXT.vocab.freqs

train_dl = torchtext.data.Iterator(
    train_ds, batch_size=batch_size, train=True)

# parameter
VOCAB_SIZE = len(TEXT.vocab.freqs)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
PAD_IDX = 1
GAMMA = 0.99
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = DQN(TEXT.vocab.vectors, VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS,
                        FILTER_SIZES, PAD_IDX)

target_model = DQN(TEXT.vocab.vectors, VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS,
                        FILTER_SIZES, PAD_IDX)

model = model.to(device)
target_model = model.to(device)
batch_size = 32

# 読み込んだ内容に対して行う処理を定義
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                            init_token="<cls>", eos_token="<eos>")

LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

target_model.load_state_dict(model.state_dict())


# 最適化手法
learning_rate = 2.5e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

num_epochs = 100
TARGET_UPDATE_FREQ = 10
# dataloaders_dict = {'train': train_dl, 'val':val_dl}
dataloaders_dict = {'train': train_dl}

print('----start----')

torch.backends.cudnn.benchmark = True

for epoch in range(num_epochs):
    epi_rewards = []
    neutrals = []
    buys = []

    # update target_model
    if epoch % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())

    for batch in (dataloaders_dict['train']):
        # curr_q
        states = batch.Text1[0].to(device)
        next_states = batch.Text2[0].to(device)
        rewards = batch.Label.to(device)

        with torch.no_grad():
            actions = torch.argmax(model(states), 1)
            actions = torch.where(torch.randn(len(states)).to(device) >= 0,
                                  actions,
                                  (actions + 1) % 2)

            selected_actions = actions.detach().cpu().numpy()
            actions = actions.view(-1, 1)

        epi_rewards.append((selected_actions * rewards.detach().cpu().numpy()).sum())
        neutrals.append(len(selected_actions[selected_actions == 0]))
        buys.append(len(selected_actions[selected_actions == 1]))

        curr_q = model(states).gather(1, actions).squeeze(dim=1)

        # target_q
        with torch.no_grad():

            next_actions = torch.argmax(model(next_states), 1).view(-1, 1)

            next_q = target_model(next_states).gather(1, next_actions)
            target_q = rewards.view(-1, 1) + (GAMMA * next_q)

        loss = torch.mean((target_q - curr_q) ** 2)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    print('--------------------')
    print('epoch:', epoch)
    print('loss:', loss.item())
    print('epi_reward:', sum(epi_rewards))
    print('neutrals:', sum(neutrals), '  buys:', sum(buys))
