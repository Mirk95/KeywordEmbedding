import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    stem_text

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


def tokenize_word(word, stem=False):
    # Lowercase
    word = word.lower()
    # Remove punctuation
    word = remove_punctuations(word)
    if stem:
        # Apply stemming
        word = stemSentence(word)
    return word


def tokenize_sentence_old(sentence, stem=False):
    sentence = sentence.strip()
    # Lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = remove_punctuations(sentence)
    if stem:
        # Apply stemming
        sentence = stemSentence(sentence)
    return sentence


def tokenize_sentence(sentence, stem=True):
    if stem:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, stem_text]
    else:
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]

    return ' '.join(preprocess_string(sentence, CUSTOM_FILTERS))


def tokenize_dataset(df, stem=False):
    """
    Dataset tokenization:
    - lowercase
    - space separation between tokens
    - remove punctuation
    - (optional) stemming
    """
    # df = pd.read_csv(input_file)
    # Remove the Dataframe columns containing all Nan values
    df = df.dropna(how='all', axis=1)

    # Apply tokenize_sentence
    df = df.applymap(lambda s: tokenize_sentence(s, stem=stem) if isinstance(s, str) else s)

    return df


if __name__ == '__main__':
    input_file = 'pipeline/datasets/title.csv'
    df = pd.read_csv(input_file, quotechar='"', error_bad_lines=False)
    new_df = tokenize_dataset(df, stem=True)
    s = tokenize_sentence("A Study in Red: The Secret Journal of Jack the Ripper", stem=True)
    w = tokenize_word("Word", stem=True)
    print(df.head(10))
    print(s)
    print(w)
