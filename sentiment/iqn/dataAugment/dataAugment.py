from dataAugment.word_embs import WordEmbsAug

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)
        return text

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomSwap:
    def __init__(self, vectors, aug_p=0.5, stopwords=[]):
        self.swap_aug = WordEmbsAug(model=vectors, action='swap', 
                  stopwords=stopwords, aug_p=aug_p)

    def __call__(self, text):
        text_transformed = self.swap_aug.augment(text)
        return text_transformed
    
class RandomInsert:
    def __init__(self, vectors, aug_p=0.5, stopwords=[]):
        self.swap_aug = WordEmbsAug(model=vectors, action='insert', 
                  stopwords=stopwords, aug_p=aug_p)

    def __call__(self, text):
        text_transformed = self.swap_aug.augment(text)
        return text_transformed

class RandomSubstitute:
    def __init__(self, vectors, aug_p=0.5, stopwords=[]):
        self.swap_aug = WordEmbsAug(model=vectors, action='substitute', 
                  stopwords=stopwords, aug_p=aug_p)

    def __call__(self, text):
        text_transformed = self.swap_aug.augment(text)
        return text_transformed

class RandomDelete:
    def __init__(self, vectors, aug_p=0.3, stopwords=[]):
        self.swap_aug = WordEmbsAug(model=vectors, action='delete', 
                  stopwords=stopwords, aug_p=aug_p)

    def __call__(self, text):
        text_transformed = self.swap_aug.augment(text)
        return text_transformed
