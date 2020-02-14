"""
    Augmenter that apply operation to textual input based on word embeddings.
"""
import numpy as np
from word_augment import WordAugmenter
# from utils.action import Action
# from model.word2vec import Word2vec
# from model.fasttext import Fasttext
# import utils.normalization as normalization


WORD_EMBS_MODELS = {}


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

    def insert(self, data):
        tokens = self.tokenize(data)
        results = tokens.copy()

        aug_idexes = self._get_random_aug_idxes(tokens)

        if aug_idexes is None:
            return data
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            new_word = self.model.itos[
                np.random.randint(len(self.model))]
            results.insert(aug_idx, new_word)
        return self.reverse_tokenizer(results)

    def substitute(self, data):
        tokens = self.tokenize(data)
        results = tokens.copy()

        aug_idexes = self._get_aug_idxes(tokens)
        if aug_idexes is None:
            return data

        for aug_idx in aug_idexes:
            original_word = results[aug_idx]
            candidate_words = self.predict(original_word, n=1)
            substitute_word = self.sample(candidate_words, 1)[0]

            results[aug_idx] = substitute_word

        return self.reverse_tokenizer(results)

    def swap(self, data):
        tokens = self.tokenize(data)
        results = tokens.copy()
    
        if len(tokens) < 2:
            return data

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return data
        aug_idexes.sort(reverse=True)

        results[aug_idexes[0]], results[aug_idexes[1]] = \
                results[aug_idexes[1]], results[aug_idexes[0]]

        return self.reverse_tokenizer(results)

    def delete(self, data):
        tokens = self.tokenize(data)
        results = tokens.copy()

        aug_idxes = self._get_random_aug_idxes(tokens)
        aug_idxes.sort(reverse=True)

        # https://github.com/makcedward/nlpaug/issues/76
        if len(tokens) < 2:
            return data

        for aug_idx in aug_idxes:
            original_token = results[aug_idx]
            del results[aug_idx]
            if aug_idx == 0:
                results[0] = self.align_capitalization(original_token, results[0])

        return self.reverse_tokenizer(results)

    def predict(self, word, n=1):
        # 単語のベクトル
        source_id = self.word2idx(word)
        source_vector = self.word2vector(word)

        # 類似度の計算
        scores = np.dot(self.normalized_vectors, source_vector)  # TODO: very slow.

        target_ids = np.argpartition(-scores, self.top_k+2)[:self.top_k+2]  # TODO: slow.
        
        target_words = [self.idx2word(idx) for idx in target_ids if 
            idx != source_id and self.idx2word(idx).lower() != word.lower()]

        return target_words[:self.top_k]

    def standard_norm(self, data):
        means = data.mean(axis =1)
        stds = data.std(axis= 1, ddof=1)
        data = (data - means[:, np.newaxis]) / stds[:, np.newaxis]
        return np.nan_to_num(data)
