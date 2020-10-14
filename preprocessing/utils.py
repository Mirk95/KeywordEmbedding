import os
import nltk
import argparse

def check_nltk_library():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='pipeline/nlp/')
        add_nltk_path('pipeline/nlp/')

def add_nltk_path(path):
    nltk.data.path.append(path)


def get_wrapper_arguments():
    parser = argparse.ArgumentParser(description='Run embedding creation')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='dataset to analyze and use to generate token embeddings')

    parser.add_argument('--sep',
                        type=str,
                        default=',',
                        help='table separator to analyze input dataset')

    parser.add_argument('--dbms',
                        action='store_true',
                        help='if file is a directory with multiple dataset')

    args = parser.parse_args()
    if args.dbms and not os.path.isdir(args.file):
        raise ValueError('{} is not a valid directory'.format(args.file))
    if not args.dbms and not os.path.isfile(args.file):
        raise ValueError('{} is not a valid file'.format(args.file))

    return args