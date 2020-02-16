from word_embs import WordEmbsAug
from torchtext.vocab import Vectors
import torch
# import MeCab

japanese_vectors = Vectors(name='../../data/news/cc.ja.300.vec')

# insert_aug = WordEmbsAug(model=japanese_vectors, action="insert",
#                   stopwords=['<cls>', '<eos>', '<sep>'])
# substitute_aug = WordEmbsAug(model=japanese_vectors, action='substitute', 
#                   stopwords=['<cls>', '<eos>', '<sep>'])
# swap_aug = WordEmbsAug(model=japanese_vectors, action='swap', 
#                   stopwords=['<cls>', '<eos>', '<sep>'])
# del_aug = WordEmbsAug(model=japanese_vectors, action='delete', 
#                   stopwords=['<cls>', '<eos>', '<sep>'])

# text = "<cls> <company> 子会社 吸い殻 再生紙 作業 自動化 効率 0倍 <eos>"
# print("Original:")
# print(text)

# """insert"""
# augmented_text = insert_aug.augment(text)
# print("Insert Text:")
# print(augmented_text)

# """ substitute """
# augmented_text = substitute_aug.augment(text)
# print("Substitute Text:")
# print(augmented_text)

# """ swap """
# augmented_text = swap_aug.augment(text)
# print("Swap Text:")
# print(augmented_text)

# """ delete """
# augmented_text = del_aug.augment(text)
# print("Delete Text:")
# print(augmented_text)

