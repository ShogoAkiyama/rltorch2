import numpy as np
from model.word_embeddings import WordEmbeddings


class Fasttext(WordEmbeddings):
    # https://arxiv.org/pdf/1712.09405.pdf,
    def __init__(self, top_k=100, skip_check=False):
        super().__init__(top_k, skip_check)
