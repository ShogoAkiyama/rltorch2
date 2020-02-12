from word_embs import WordEmbsAug
from torchtext.vocab import Vectors


text = "決算 星取表 <company> ● ｻﾌﾟﾗｲｽﾞﾚｼｵ N 0年0月期 <span> <company> 前期" \
       " 連結 最終 益 0億 円 前々 期 0億 円 黒字 <span> <company> 安い 0年0月期 純利益 0% 減"
print(text)

japanese_vectors = Vectors(name='../data/news/cc.ja.300.vec')

"""insert"""
aug = WordEmbsAug(model=japanese_vectors, model_type='fasttext', action='insert')
augmented_text = aug.augment(text)
print("Insert Text:")
print(augmented_text)

""" substitute """
aug = WordEmbsAug(model=japanese_vectors, model_type='fasttext', action='substitute')
augmented_text = aug.augment(text)
print("Substitute Text:")
print(augmented_text)

""" swap """
aug = WordEmbsAug(model=japanese_vectors, model_type='fasttext', action='swap')
augmented_text = aug.augment(text)
print("Swap Text:")
print(augmented_text)

""" delete """
aug = WordEmbsAug(model=japanese_vectors, model_type='fasttext', action='delete')
augmented_text = aug.augment(text)
print("Delete Text:")
print(augmented_text)

""" bert model """

