from word_embs import WordEmbsAug
from torchtext.vocab import Vectors


text = "今日 は 良い 天気 です"
# model_type: word2vec, glove or fasttext
# aug = WordEmbsAug(
#     model_type='fasttext', model_path='../data/news/cc.ja.300.vec',
#     action="insert")


japanese_vectors = Vectors(name='../data/news/cc.ja.300.vec')

"""insert"""
# augmented_text = aug.augment(text)
# print("Original:")
# print(text)
# print("Augmented Text:")
# print(augmented_text)

""" substitute """
aug = WordEmbsAug(model=japanese_vectors, model_type='fasttext', action='substitute')
augmented_text = aug.augment(text)
print("Augmented Text:")
print(augmented_text)

""" swap """

