# Keyword Embedding Framework

Master Thesis in Computer Engineering - *Department of Engineering "Enzo Ferrari"* - April 2021

Title: **"Study of Word Embeddings techniques finalized to Keyword Search"**

Supervisor: Prof. *Francesco Guerra*

Advisors: Ing. *Matteo Paganelli*, Ing. *Francesco Del Buono*

# Related Works

At the beginning of our work, we started to study the main word embedding techniques known in Literature (such as Word2Vec, GloVe, Naive Bayes, BERT, ELMo, GPT etc.) and to understanding how to use them for structured data taken from databases.
Analyzing the various approaches proposed by numerous data scientists from all over the world, we focused our attention on two frameworks:
* [EmbDI by R. Cappuzzo, P. Papotti, and S. Thirumuruganathan](https://gitlab.eurecom.fr/cappuzzo/embdi)

* [RETRO by M. Günther, P. Oehme, M. Thiele, and W. Lehner](https://github.com/guenthermi/postgres-retrofit)

## EmbDI
>"EmbDI is a Python library developed to perform Entity Resolution (ER) and Schema Matching (SM) tasks by employing word embeddings".

EmbDI is a framework which consists of three major components, as depicted in the following figure, taken from the authors' paper:
<p align="center">
  <img src="https://github.com/Mirk95/KeywordEmbedding/blob/master/images/EmbDI.png">
</p>

1. In the **Graph Construction** stage, the relational dataset is processed and transformed into a compact tripartite graph that encodes various relationships inherent in it. Tuple and attribute ids are treated as first class citizens. In detail, a heterogeneous graph with three types of nodes is obtained: *Token* nodes correspond to information found in the dataset (i.e. the content of each cell in the relation), *Record Id* nodes (RIDs) represent each tuple in the dataset, *Column Id* nodes (CIDs) represent each column/attribute. These nodes are connected by edges according to the structural relationships in the schema.
2. Given this graph, the next step is **Sentence Construction** through the use of biased random walks. These walks are carefully constructed to avoid common issues such as rare words and imbalance in vocabulary sizes. This produces as output a series of sentences. The use of graphs and random walks allows to have a richer and more diverse set of neighborhoods than what would be possible by encoding a tuple as a single sentence.
3. Finally, in **Embedding Construction**, the corpus of sentences is passed to an algorithm for learning word embeddings. The approach is totally agnostic to the actual word embedding algorithm used. Depending on available external information, some optimizations can be performed to the graph and the workflow to improve the embeddings’ quality.

## RETRO
> "RETRO is a framework that provides tools to automatically extract text values from a PostgreSQL database, represent those text values by a continuous vector representation using a word embedding model. In order to incorporate semantic knowledge from the database into the representation, it extracts additional relational information from the database and uses this knowledge to refine the embeddings by a relational retrofitting method. The resulting embeddings can then be used to perform machine learning tasks".

The relational retrofitting approach is shown in the folloqing figure, taken from the authors' paper:
<p align="center">
  <img src="https://github.com/Mirk95/KeywordEmbedding/blob/master/images/RETRO.png">
</p>

1. The input of the overall process consists of a database (*Step 1a*) as well as a given word embedding W<sub>0</sub>, either pre-trained or self-trained for a specific domain (*Step 1b*).
2. In a first step (*Step 2a*), all database text values are extracted together with the information in which column they appeared, capturing their “category". Further the relationships between all text values are extracted, e.g. PrimaryKey-ForeignKey relations. Multi-word phrases for which word embeddings are available are preserved by a specific tokenization approach. Words in the database having no counter-part in the given embedding W<sub>0</sub> are initialized with a null vector. The extracted relationships (*Step 2a*) and the tokenized text values (*Step 2b*) are combined to a property graph representation (*Step 2c*).
3. The graph encodes all the relations between text values according to the given database schema. The relational retrofitting approach RETRO adapts the base word embedding representation W<sub>0</sub> using the set of relationships encoded in the graph. The result will be the retrofitted embeddings (*Step 3a*) containing vectors for all terms appearing in the input database. In addition to the core algorithm, the authors investigated an existing node embedding technique called *DeepWalk* taking the already derived graph representation (*Step 2c*) as an input. Node embedding approaches can encode relational information more accurately than retrofitting approaches which also need to maintain connection with the word representation. Since node embeddings (*Step 3b*) should perform better than retrofitted embeddings (*Step 3a*) when relational features are prevalent in a ML task in opposite to pure to textual information, the authors looked for how node embeddings can be trained on relational databases and combined with retrofitted embeddings (*Step 3c*).

# Implementation Details
The purpose of this Master Thesis is to understand which are the approaches that are used in Literature to create the most efficient vector representations for structured data and try to use these embeddings to perform keyword search tasks on databases. 
In particular, we focused our attention on the [Coffman & Weaver's benchmark](https://dataverse.lib.virginia.edu/dataset.xhtml?persistentId=doi:10.18130/V3/KEVCF8), which is widely used by numerous data scientists to evaluate database keyword search techniques.

> "The benchmark for relational keyword search is a collection of data sets, queries, and relevance assessments designed to facilitate the evaluation of systems supporting keyword search in databases. The benchmark includes three separate data sets with fifty information needs (i.e., queries) for each data set and follows the traditional approach to evaluate keyword search systems developed by the information retrieval (IR) research community".

Among the three datasets available in the benchmark, we have considered IMDB, which is a subset of the original database. It consists of six tables: cast_info, char_name, movie_info, name, role_type and title, linked by foreign key relations. 

The following figure shows an explanatory diagram of the dataset:
<p align="center">
  <img src="https://github.com/Mirk95/KeywordEmbedding/blob/master/images/DB_Schema.png">
</p>

## Project Tree
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
# References
* Cappuzzo, Riccardo and Papotti, Paolo and Thirumuruganathan, Saravanan - 2020 - [*Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks*](http://dx.doi.org/10.1145/3318464.3389742) - Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data.
* Michael Günther, Philipp Oehme, Maik Thiele, and Wolfgang Lehner - 2020 - [*Learning from Textual Data in Database Systems*](https://doi.org/10.1145/3340531.3412056) - Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). ACM, New York, NY, USA, 375–384.
* Joel Coffman and Alfred C. Weaver, [*A Framework for Evaluating Database Keyword Search Strategies*](https://dl.acm.org/doi/10.1145/1871437.1871531) - Proceedings of the 19th ACM International Conference on Information and Knowledge Management (CIKM ‘10), pp. 729–738, Toronto, Canada, pp. 729–738, October 2010