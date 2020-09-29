Comimport pandas as pd
import string

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()


def remove_punctuations(sentence):
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    return sentence

def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def tokenize_dataset(input_file, stemming=False):
    '''
    Dataset tokenization:
    - lowercase
    - space separation between tokens
    - remove puntuation
    - (optional) stemming
    '''
    df = pd.read_csv(input_file)
    # Remove the Dataframe columns containing all Nan values
    df = df.dropna(how='all', axis=1)
    # Lowercase the Dataframe
    df = df.applymap(lambda s:s.lower() if isinstance(s, str) else s)
    # Remove the punctuation
    df = df.applymap(lambda s:remove_punctuations(s) if isinstance(s, str) else s)
    with stemming == True:
        df = df.applymap(lambda s:stemSentence(s) if isinstance(s, str) else s)
    return df


def tokenize_sentence():
    pass


def tokenize_word():
    pass


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    df = tokenize_dataset(input_file)