import os
import argparse
import shutil

import torch
from torchtext.vocab import Vectors

from utils import tokenizer_with_preprocessing, MyDataset
from trainer import Trainer


NEWS_PATH = os.path.join('..', 'data', 'news')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--evaluation_freq', type=int, default=10)
    parser.add_argument('--network_save_freq', type=int, default=100)
    parser.add_argument('--num_actions', type=int, default=1)

    parser.add_argument('--min_freq', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--n_filters', type=int, default=50)
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('--pad_idx', type=list, default=1)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=2.5e-5)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--rnn', action='store_true')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)

    # IQN
    parser.add_argument('--num_quantile', type=int, default=32)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    # ログファイルを入れる
    log_dir = os.path.join('.', 'logs', str(args.num_quantile) + '_' + str(args.gamma))
    summary_dir = os.path.join(log_dir, 'summary')

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    else:
        shutil.rmtree(summary_dir)
        os.makedirs(summary_dir)

    japanese_vectors = Vectors(name='../data/news/cc.ja.300.vec')

    # Create Dataset
    train_ds = MyDataset(
        path=os.path.join('..', 'data', 'news', 'text_train.tsv'),
        specials=['<company>', '<organization>', '<person>', '<location>'],
        max_len=args.max_length,
        vectors=japanese_vectors,
        min_freq=args.min_freq,
        phase='train'
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size = args.batch_size, 
        shuffle = True, 
        num_workers = 3
    )

    text_vectors = train_ds.vocab.vectors
    vocab_size = len(train_ds.vocab.vectors)

    trainer = Trainer(args, text_vectors, vocab_size, train_dl)
    trainer.run()
