import os
import argparse
import torchtext
import torch

from utils import tokenizer_with_preprocessing
from trainer import Trainer

NEWS_PATH = os.path.join('..', 'data', 'news')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--target_update_freq', type=int, default=100)

    parser.add_argument('--min_freq', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('--pad_idx', type=list, default=1)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    # 読み込んだ内容に対して行う処理を定義
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                                use_vocab=True, lower=True, include_lengths=True,
                                batch_first=True, fix_length=args.max_length,
                                init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float)

    train_ds = torchtext.data.TabularDataset.splits(
        path=NEWS_PATH, train='text_train.tsv',
        format='tsv',
        fields=[('Text1', TEXT), ('Text2', TEXT), ('Label', LABEL)])
    train_ds = train_ds[0]

    TEXT.build_vocab(train_ds, min_freq=10)
    TEXT.vocab.freqs

    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=args.batch_size, train=True)

    trainer = Trainer(args, TEXT, train_dl)
    trainer.run()

