from word_embs import WordEmbsAug
from torchtext.vocab import Vectors
# import MeCab


text = "<cls> a i u e o <eos>"
print("Original:")
print(text)
# model_type: word2vec, glove or fasttext
# aug = WordEmbsAug(
#     model_type='fasttext', model_path='../data/news/cc.ja.300.vec',
#     action="insert")

japanese_vectors = Vectors(name='../data/news/cc.ja.300.vec')

# # """insert"""
# aug = WordEmbsAug(model=japanese_vectors, action='insert', stopwords=['<cls>', '<eos>', '<sep>'])
# augmented_text = aug.augment(text)
# print("Augmented Text:")
# print(augmented_text)

# # """ substitute """
# aug = WordEmbsAug(model=japanese_vectors, action='substitute', stopwords=['<cls>', '<eos>', '<sep>'])
# augmented_text = aug.augment(text)
# print("Augmented Text:")
# print(augmented_text)

# """ swap """
aug = WordEmbsAug(model=japanese_vectors, action='swap', stopwords=['<cls>', '<eos>', '<sep>'])
augmented_text = aug.augment(text)
print("Augmented Text:")
print(augmented_text)
