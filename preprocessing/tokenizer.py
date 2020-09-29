import pandas as pd
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
    return "".join(stem_sentence).rstrip()


def tokenize_word(word):
    # Lowercase
    word = word.lower()
    # Don't remove punctuation because in a single word it's not useful...
    # Don't apply stemming because in a single word it's not useful...
    return word


def tokenize_sentence(sentence, stem=False):
    # Lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = remove_punctuations(sentence)
    if stem:
        # Apply stemming
        sentence = stemSentence(sentence)
    return sentence


def tokenize_dataset(input_file, stem=False):
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
    # Apply tokenize_sentence
    df = df.applymap(lambda s:tokenize_sentence(s, stem=stem) if isinstance(s, str) else s)
    return df


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    df = tokenize_dataset(input_file, stem=False)
    s = tokenize_sentence("Hello!!!! Let's programming!!!", stem=True)
    w = tokenize_word("Word")
    print(df.head(10))
    print(s)
    print(w)