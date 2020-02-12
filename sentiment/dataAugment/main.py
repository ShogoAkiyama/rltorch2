import word_augment as naw


text = "今日 は 良い 天気 です"
# model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path="/Users/shogoakiyama/Desktop/rltorch2/sentiment/data/news/cc.ja.300.bin",
    action="insert")

augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

# model_type: word2vec, glove or fasttext
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path="/Users/shogoakiyama/Desktop/rltorch2/sentiment/data/news/cc.ja.300.bin",
    action="substitute")
augmented_text = aug.augment(text)
print("Augmented Text:")
print(augmented_text)