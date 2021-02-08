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
>EmbDI is a Python library developed to perform Entity Resolution (ER) and Schema Matching (SM) tasks by employing word embeddings.

EmbDI is a framework which consists of three major components, as depited in the following figure, taken from authors' paper:
<p align="center">
  <img src="https://github.com/Mirk95/KeywordEmbedding/images/EmbDI.png">
</p>
1. In the Graph Construction stage, the authors process the relational dataset and transform it to a compact tripartite graph that encodes various relationships inherent in it. Tuple and attribute ids are treated as first class citizens.
2. Given this graph, the next step is *Sentence Construction* through the use of biased random walks. These walks are carefully constructed to avoid common issues such as rare words and imbalance in vocabulary sizes. This produces as output a series of sentences.

### RETRO
>RETRO is a framework that provides tools to automatically extract text values from a PostgreSQL database, represent those text values by a continuous vector representation using a word embedding model. In order to incorporate semantic knowledge from the database into the representation, it extracts additional relational information from the database and uses this knowledge to refine the embeddings by a relational retrofitting method. The resulting embeddings can then be used to perform machine learning tasks.

## Implementation Details
### Project Tree
```
.
├── Base
│   ├── base.py
│   ├── __init__.py
├── dbms2graph.py
├── EmbDI
│   ├── embeddings.py
│   ├── graph.py
│   ├── logging.py
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
|   |       ├── ...
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
│   ├── tokenizer.py
│   └── utils.py
├── query_embedding.py
├── query_embedding_valutation.py
├── README.md
├── requirements.txt
├── RETRO
│   ├── gml2json.py
│   ├── graph_generation.py
│   ├── group_extraction.py
│   ├── matrix_retrofit.py
│   └── retro_utils.py
├── RETRO_Numeric
│   ├── encoding_utils.py
│   ├── gml2json.py
│   ├── graph_generation.py
│   ├── group_extraction.py
│   ├── matrix_retrofit.py
│   └── retro_utils.py
└── wrapper
    ├── base_wrapper.py
    ├── edgelist.py
    ├── embdi_wrapper.py
    ├── __init__.py
    ├── retro_numeric_wrapper.py
    └── retro_wrapper.py
```
## References
* Cappuzzo, Riccardo and Papotti, Paolo and Thirumuruganathan, Saravanan - 2020 - [*Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks*](http://dx.doi.org/10.1145/3318464.3389742) - Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data.
* Michael Günther, Philipp Oehme, Maik Thiele, and Wolfgang Lehner - 2020 - [*Learning from Textual Data in Database Systems*](https://doi.org/10.1145/3340531.3412056) - In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). ACM, New York, NY, USA, 375–384.