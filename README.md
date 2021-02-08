# Keyword Embedding Framework

Master Thesis in Computer Engineering - *Department of Engineering "Enzo Ferrari"* - April 2021

Title: **"Study of Word Embeddings techniques finalized to Keyword Search"**

Supervisor: Prof. *Francesco Guerra*

Advisors: Ing. *Matteo Paganelli*, Ing. *Francesco Del Buono*

## Related Works

At the beginning of our work, we started to study the main word embedding techniques known in Literature (such as Word2Vec, GloVe, Naive Bayes, BERT, ELMo, GPT etc.) and to understanding how to use them for structured data taken from databases.
Analyzing the various approaches proposed by numerous data scientists from all over the world, we focused our attention on two frameworks:
* [EmbDI](https://gitlab.eurecom.fr/cappuzzo/embdi)

* [RETRO](https://github.com/guenthermi/postgres-retrofit)

### EmbDI
>EmbDI is a Python library developed to perform Entity Resolution (ER) and Schema Matching (SM) tasks by employing
word embeddings.

### RETRO
>RETRO is a framework that provides tools to automatically extract text values from a PostgreSQL database, represent those text values by a continuous vector representation using a word embedding model. In order to incorporate semantic knowledge from the database into the representation, it extracts additional relational information from the database and uses this knowledge to refine the embeddings by a relational retrofitting method. The resulting embeddings can then be used to perform machine learning tasks.

### Tree
```
.
├── Base
│   ├── base.py
│   ├── __init__.py
│   └── __pycache__
│       ├── base.cpython-38.pyc
│       └── __init__.cpython-38.pyc
├── dbms2graph.py
├── EmbDI
│   ├── embeddings.py
│   ├── graph.py
│   ├── logging.py
│   ├── __pycache__
│   │   ├── embeddings.cpython-38.pyc
│   │   ├── graph.cpython-38.pyc
│   │   ├── logging.cpython-38.pyc
│   │   ├── sentence_generation_strategies.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── sentence_generation_strategies.py
│   └── utils.py
├── pipeline
│   ├── columns
│   │   └── db_columns.json
│   ├── datasets
│   │   ├── cast_info.csv
│   │   ├── char_name.csv
│   │   ├── movie_info.csv
│   │   ├── name.csv
│   │   ├── role_type.csv
│   │   └── title.csv
│   ├── embeddings
│   │   ├── base_fasttext__datasets.emb
│   │   ├── base_word2vec_CBOW__datasets.emb
│   │   └── retro__datasets.emb
│   ├── nlp
│   │   └── __init__.py
│   ├── queries
│   │   └── IMDB
│   │       ├── 001.txt
│   │       ├── 002.txt
│   │       ├── 003.txt
│   │       ├── 004.txt
│   │       ├── 005.txt
│   │       ├── 006.txt
│   │       ├── 007.txt
│   │       ├── 008.txt
│   │       ├── 009.txt
│   │       ├── 010.txt
│   │       ├── 011.txt
│   │       ├── 012.txt
│   │       ├── 013.txt
│   │       ├── 014.txt
│   │       ├── 015.txt
│   │       ├── 016.txt
│   │       ├── 017.txt
│   │       ├── 018.txt
│   │       ├── 019.txt
│   │       ├── 020.txt
│   │       ├── 021.txt
│   │       ├── 022.txt
│   │       ├── 023.txt
│   │       ├── 024.txt
│   │       ├── 025.txt
│   │       ├── 026.txt
│   │       ├── 027.txt
│   │       ├── 028.txt
│   │       ├── 029.txt
│   │       ├── 030.txt
│   │       ├── 031.txt
│   │       ├── 032.txt
│   │       ├── 033.txt
│   │       ├── 034.txt
│   │       ├── 035.txt
│   │       ├── 036.txt
│   │       ├── 037.txt
│   │       ├── 038.txt
│   │       ├── 039.txt
│   │       ├── 040.txt
│   │       ├── 041.txt
│   │       ├── 042.txt
│   │       ├── 043.txt
│   │       ├── 044.txt
│   │       ├── 045.txt
│   │       ├── 046.txt
│   │       ├── 047.txt
│   │       ├── 048.txt
│   │       ├── 049.txt
│   │       ├── 050.txt
│   │       ├── README
│   │       └── topics.txt
│   ├── query_emb_results
│   │   ├── baseword2vecCBOW_cnsearch_base.pickle
│   │   ├── baseword2vecCBOW_singletosingle.pickle
│   │   ├── baseword2vecskipgram_cnsearch_base.pickle
│   │   ├── baseword2vecskipgram_singletosingle.pickle
│   │   ├── embdi_cnsearch_base.pickle
│   │   └── embdi_singletosingle.pickle
│   ├── schemas
│   │   ├── db_schema.json
│   │   └── google_vecs_schema.json
│   ├── vectors
│   │   ├── GoogleNews-vectors-negative300.bin.gz
│   │   └── google_vecs.csv
│   └── walks
│       ├── base_datasets.walks
│       ├── base_name.walks
│       └── name.walks
├── preprocessing
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── tokenizer.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── tokenizer.py
│   └── utils.py
├── __pycache__
│   └── dbms2graph.cpython-38.pyc
├── query_embedding.py
├── query_embedding_valutation.py
├── README.md
├── requirements.txt
├── RETRO
│   ├── gml2json.py
│   ├── graph_generation.py
│   ├── group_extraction.py
│   ├── matrix_retrofit.py
│   ├── __pycache__
│   │   ├── gml2json.cpython-38.pyc
│   │   ├── graph_generation.cpython-38.pyc
│   │   ├── group_extraction.cpython-38.pyc
│   │   ├── matrix_retrofit.cpython-38.pyc
│   │   ├── retro_utils.cpython-38.pyc
│   │   └── retro_wrapper.cpython-38.pyc
│   └── retro_utils.py
├── RETRO_Numeric
│   ├── encoding_utils.py
│   ├── gml2json.py
│   ├── graph_generation.py
│   ├── group_extraction.py
│   ├── matrix_retrofit.py
│   ├── __pycache__
│   │   ├── encoding_utils.cpython-38.pyc
│   │   ├── gml2json.cpython-38.pyc
│   │   ├── graph_generation.cpython-38.pyc
│   │   ├── group_extraction.cpython-38.pyc
│   │   ├── matrix_retrofit.cpython-38.pyc
│   │   └── retro_utils.cpython-38.pyc
│   └── retro_utils.py
└── wrapper
    ├── base_wrapper.py
    ├── edgelist.py
    ├── embdi_wrapper.py
    ├── __init__.py
    ├── __pycache__
    │   ├── base_wrapper.cpython-38.pyc
    │   ├── edgelist.cpython-38.pyc
    │   ├── embdi_wrapper.cpython-38.pyc
    │   ├── __init__.cpython-38.pyc
    │   ├── retro_numeric_wrapper.cpython-38.pyc
    │   └── retro_wrapper.cpython-38.pyc
    ├── retro_numeric_wrapper.py
    └── retro_wrapper.py
```
## References
* Cappuzzo, Riccardo and Papotti, Paolo and Thirumuruganathan, Saravanan - 2020 - [*Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks*](http://dx.doi.org/10.1145/3318464.3389742) - Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data.
* Michael Günther, Philipp Oehme, Maik Thiele, and Wolfgang Lehner - 2020 - [*Learning from Textual Data in Database Systems*](https://doi.org/10.1145/3340531.3412056) - In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). ACM, New York, NY, USA, 375–384.