import numpy as np
from dataAugment.word_augment import WordAugmenter
import torch


class WordEmbsAug(WordAugmenter):
    # https://aclweb.org/anthology/D15-1306, https://arxiv.org/pdf/1804.07998.pdf, https://arxiv.org/pdf/1509.01626.pdf
    # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf

    def __init__(self, model_path='.', model=None, action='substitute',
                 aug_min=1, aug_max=10, aug_p=0.3, top_k=100, n_gram_separator='_',
                 stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu',
            stopwords_regex=stopwords_regex)

        self.top_k = top_k
        self.model = model

        self.words = list(self.model.itos)
        self.normalized_vectors = \
            self.standard_norm(self.model.vectors.numpy())

        if action == 'substitute':
            self.scores = np.dot(self.normalized_vectors, self.normalized_vectors.T)

    def align_capitalization(self, src_token, dest_token):
        if self.get_word_case(src_token) == 'capitalize' and self.get_word_case(dest_token) == 'lower':
            return dest_token.capitalize()
        return dest_token

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:
            # Some words do not come with vector. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in self.words:
                results.append(token_idx)

        return results

    def substitute(self, data):
        # tokens = self.tokenize(data)
        tokens = data.copy()
        results = tokens

        aug_idexes = self._get_random_aug_idxes(tokens)

        if aug_idexes is None:
            return data

        for aug_idx in aug_idexes:
            original_word = results[aug_idx]
            candidate_words = self.predict(original_word, n=1)

            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        # return self.reverse_tokenizer(results)
        return results

    def swap(self, data):
        # tokens = self.tokenize(data)
        tokens = data.copy()

        results = tokens
    
        if len(tokens) < 2:
            return data

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return data
        elif len(aug_idexes) < 2:
            return data

        for idx in range(len(aug_idexes)-1):
            results[aug_idexes[idx]], results[aug_idexes[idx+1]] = \
                results[aug_idexes[idx+1]], results[aug_idexes[idx]]

        results[aug_idexes[0]], results[aug_idexes[1]] = \
                results[aug_idexes[1]], results[aug_idexes[0]]

        # return self.reverse_tokenizer(results)
        return results

    def predict(self, idx, n=1):
        # 単語のベクトル
        source_id = idx   #self.word2idx(word)
        source_vector = self.idx2vector(idx)   #self.word2vector(word)

        # 類似度の計算
        # scores = np.dot(self.normalized_vectors, source_vector)  # TODO: very slow.

        target_ids = np.argpartition(-self.scores[idx], self.top_k+2)[:self.top_k+2]  # TODO: slow.
        
        target_idx = [idx for idx in target_ids if 
                        (idx != source_id)]

        return target_idx[:self.top_k]

    def standard_norm(self, data):
        means = data.mean(axis =1)
        stds = data.std(axis= 1, ddof=1)
        data = (data - means[:, np.newaxis]) / (stds[:, np.newaxis] + 1e-10)
        return np.nan_to_num(data)


class Fasttext:
    # https://arxiv.org/pdf/1712.09405.pdf,
    def __init__(self, top_k=100, skip_check=False):
        self.top_k = top_k
        self.skip_check = skip_check
        self.emb_size = 0
        self.vocab_size = 0
        self.embs = {}
        self.vectors = []
        self.normalized_vectors = None
