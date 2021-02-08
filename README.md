# Keyword Embedding Framework

*Master Thesis in Computer Engineering - Engineering Department "Enzo Ferrari" - April 2021*

Title: **"Study of Word Embeddings techniques finalized to Keyword Search"**

Supervisor: *Prof. Francesco Guerra*

Advisors: *Ing. Matteo Paganelli, Ing. Francesco Del Buono*

## Related Works

At the beginning of our work we start studying the main word embedding techniques known in Literature (like Word2Vec, GloVe, Naive Bayes, BERT, ELMo, GPT etc.) and understanding how to use them for structured data taken from databases.
Analyzing the various approaches proposed by numerous data scientists from all over the world, we focused our attention on two frameworks:
* [EmbDI](https://gitlab.eurecom.fr/cappuzzo/embdi)

* [RETRO](https://github.com/guenthermi/postgres-retrofit)

### EmbDI
>EmbDI is a Python library developed to perform Entity Resolution (ER) and Schema Matching (SM) tasks by employing
word embeddings.

### RETRO
>RETRO is a framework that provides tools to automatically extract text values from a PostgreSQL database, represent those text values by a continuous vector representation using a word embedding model. In order to incorporate semantic knowledge from the database into the representation, it extracts additional relational information from the database and uses this knowledge to refine the embeddings by a relational retrofitting method. The resulting embeddings can then be used to perform machine learning tasks.
## References
* Cappuzzo, Riccardo and Papotti, Paolo and Thirumuruganathan, Saravanan - 2020 - [*Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks*](http://dx.doi.org/10.1145/3318464.3389742) - Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data.
* Michael Günther, Philipp Oehme, Maik Thiele, and Wolfgang Lehner - 2020 - [*Learning from Textual Data in Database Systems*](https://doi.org/10.1145/3340531.3412056) - In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). ACM, New York, NY, USA, 375–384.