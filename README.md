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
The purpose of this Master Thesis is to understand which are the best approaches that are used in Literature to create the most efficient vector representations for structured data and try to use these embeddings to perform keyword search tasks on databases. 

## The Dataset
We focused our attention on the [Coffman & Weaver's benchmark](https://dataverse.lib.virginia.edu/dataset.xhtml?persistentId=doi:10.18130/V3/KEVCF8), which is widely used by numerous data scientists to evaluate database keyword search techniques.

> "The benchmark for relational keyword search is a collection of data sets, queries, and relevance assessments designed to facilitate the evaluation of systems supporting keyword search in databases. The benchmark includes three separate data sets with fifty information needs (i.e., queries) for each data set and follows the traditional approach to evaluate keyword search systems developed by the information retrieval (IR) research community".

Among the three datasets available in the benchmark (i.e., *IMDB*, *Mondial* and *Wikipedia*), we have considered ***IMDB***, which is a subset of the original database. It consists of six relations: *cast_info*, *char_name*, *movie_info*, *name*, *role_type* and *title*, linked by foreign keys. Each record in each table has a unique attribute, *search_id*, which will be used in the keyword search phase.

An explanatory diagram of the dataset with the respective primary key and foreign key relations is shown in the following figure:
<p align="center">
  <img src="https://github.com/Mirk95/KeywordEmbedding/blob/master/images/DB_Schema.png">
</p>


* The *title* relation contains 181.706 tuples, each representing a particular movie with related information, such as production year, season number, episode number and many others.
* The *name* relation is composed by 273.034 tuples, each representing a person with related information.
* The *char_name* relation consists of 206.952 tuples, each representing the name of a character from a given movie.
* The *role_type* relation is made up of 11 tuples, each representing a specific role (between actor, actress, producer, writer, cinematographer, composer, costume designer, director, editor, miscellaneous crew and production designer) that a person has in a movie.
* The *movie_info* relation, consisting of 192.678 tuples, contains additional information, such as famous quotes from movies. The table has a foreign key relation with the *title* table.
* Finally, the *cast_info* relation is made up of 812.695 tuples and links a specific person who has participated in a particular film with a certain character name and a specific role in the cast through foreign key.

## EmbDI Wrapper
The first wrapper created is the one based on *EmbDI* framework, developed entirely in Python language, version 3.8.

The original *EmbDI* code has been modified to take as input not a single dataset at a time but a directory with multiple csv files inside, since the database under consideration consists of six relations. Furthermore, for each dataset a tokenization has carried out, which consists in putting words in lowercase, removing punctuation, separating the various tokens of each attribute with a space and, optionally, stemming the words.

### Graph Construction
Before the first phase, it is necessary to create a list of edges which is then used to construct the graph starting from a relational dataset.
In particular, an edgelist is extracted for each dataset and all these lists are then aggregated into a single one.
Few lines of each edgelist are shown below:
* *cast_info.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__cast_info__0,tn__396144,1.0,1.0
tn__396144,cid__id,1.0,1.0
idx__cast_info__0,tn__56514,1.0,1.0
tn__56514,cid__person_id,1.0,1.0
idx__cast_info__0,tn__104361,1.0,1.0
tn__104361,cid__movie_id,1.0,1.0
...
```
* *char_name.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__char_name__0,tn__1,1.0,1.0
tn__1,cid__id,1.0,1.0
idx__char_name__0,tt__itself,1.0,1.0
tt__itself,cid__name,1.0,1.0
idx__char_name__0,tt__i3241,1.0,1.0
tt__i3241,cid__name_pcode_nf,1.0,1.0
...
```
* *movie_info.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__movie_info__0,tn__316189,1.0,1.0
tn__316189,cid__id,1.0,1.0
idx__movie_info__0,tn__160,1.0,1.0
tn__160,cid__movie_id,1.0,1.0
idx__movie_info__0,tn__15,1.0,1.0
tn__15,cid__info_type_id,1.0,1.0
...
```
* *name.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__name__0,tn__6431,1.0,1.0
tn__6431,cid__id,1.0,1.0
idx__name__0,tt__adler robert,1.0,1.0
tt__adler robert,cid__name,1.0,1.0
idx__name__0,tt__i,1.0,1.0
tt__i,cid__imdb_index,1.0,1.0
...
```
* *role_type.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__role_type__0,tn__1,1.0,1.0
tn__1,cid__id,1.0,1.0
idx__role_type__0,tt__actor,1.0,1.0
tt__actor,cid__role,1.0,1.0
idx__role_type__1,tn__2,1.0,1.0
tn__2,cid__id,1.0,1.0
...
```
* *title.edges*
```
  3#__tn,3$__tt,5$__idx,1$__cid
idx__title__0,tn__15680,1.0,1.0
tn__15680,cid__id,1.0,1.0
idx__title__0,tt__a stranger in my arm,1.0,1.0
tt__a stranger in my arm,cid__title,1.0,1.0
idx__title__0,tn__1,1.0,1.0
tn__1,cid__kind_id,1.0,1.0
...
```

All these edgelists are put together in a single dataframe which is the
input of the graph construction phase. This step generates the graph following the specifications in configuration and returns the generated graph.

### Sentence Construction
The sentences generation phase runs through the graph using different random walks strategies. In particular, it takes as input the configuration parameters, the input dataframe and the graph generated in the previous step and returns a collection of random walks.
Applying it to the dataframe containing the edgelists of all the datasets and to the graph created in the previous phase, the sentence generation step produced *69.653.066* sentences.

```
idx__title__154514 tn__1 idx__cast_info__801453 tn__9564 idx__cast_info__367014 tn__222579 idx__cast_info__367014 tn__9564 idx__cast_info__751210 tn__1 idx__cast_info__584326 tn__2 idx__cast_info__806737 tn__190698 idx__movie_info__16304 tt__nasia idx__movie_info__16304 tt__thing idx__movie_info__149639 tt__snow idx__title__77034 tt__ii idx__name__102075 tt__m625 idx__name__209944 tt__m625 idx__name__57307 tt__julio idx__movie_info__103821 tt__us idx__movie_info__185677 tt__after_c_d_open_the_door_to_her_hous_roxann_kowalski_do_you_want_to_come_in_she_come_in_and_see_he_is_alreadi_in_the_kitchen_prepar_some_chees_and_veget_cutlet_c_d_bale_uh_i_sort_of_alreadi_did_i_figur_you_must_be_starv_so_i_just_made_us_some_chees_and_veget_au_naturel_roxann_kowalski_mayb_you_d_like_some_wine_with_your_nose_c_d_cut_a_piec_of_chees_unusu_forcefulli_roxann_kowalski_chees cid__movie_info__info tt__braver idx__movie_info__78091 tt__there idx__movie_info__41477 tt__me idx__movie_info__71645 tt__unzip idx__movie_info__37933 tt__do idx__movie_info__64352 tt__a idx__movie_info__73158 tt__cohen idx__movie_info__73150 tt__insan idx__movie_info__27472 tt__ve idx__title__161263 tt__matok idx__title__161263 tt__m3215 cid__name__name_pcode_cf tt__k3626 idx__name__183260 tt__k3626 idx__name__79721 tt__nair idx__name__38876

idx__title__57236 tn__1 idx__title__141735 tn__67688 cid__cast_info__movie_id tn__455093 idx__cast_info__368351 tn__276168 idx__cast_info__242746 tn__1 idx__cast_info__182258 tn__5864343 cid__cast_info__id tn__19782374 idx__cast_info__139653 tn__19782374 idx__cast_info__139653 tn__19782374 cid__cast_info__id tn__2080175 idx__cast_info__788262 tn__2080175 idx__cast_info__788262 tn__135564 idx__cast_info__27498 tn__2900928 cid__cast_info__id tn__14400654 idx__cast_info__652628 tn__198287 idx__name__210619 tt__costello_shaun cid__name__name tt__wansburi_penelop cid__name__name tt__pankoff cid__name__name tt__pochet cid__name__name tt__macocco_elisabeth idx__name__27639 tt__m2421 idx__name__227096 tt__m2421 idx__name__44053 tt__isabel idx__movie_info__143271 tt__it idx__movie_info__79920 tt__you idx__movie_info__196080 tt__i idx__movie_info__85233 tt__poni idx__movie_info__100151 tt__sport idx__title__66180 tn__458704 idx__movie_info__110830 tt__traci idx__movie_info__124426

idx__cast_info__594675 tn__1 idx__title__63662 tt__el idx__movie_info__184449 tt__not idx__movie_info__21535 tt__strong idx__movie_info__136626 tn__15 idx__movie_info__128098 tt__ar idx__movie_info__21888 tt__shakespear idx__movie_info__72952 tt__come idx__movie_info__43711 tt__let idx__movie_info__57186 tt__wretch idx__movie_info__58518 tt__perfectli idx__movie_info__89617 tt__it idx__movie_info__170852 tt__thei idx__title__72540 tt__live idx__movie_info__48060 tt__anyth idx__movie_info__149389 tn__440696 idx__cast_info__243616 tn__3295416 idx__cast_info__243616 tn__750840 idx__char_name__104671 tt__matis_holland idx__char_name__104671 tn__750840 idx__cast_info__243616 tn__440696 idx__cast_info__298265 tt__consult_produc idx__cast_info__769393 tt__produc idx__cast_info__83949 tn__14058270 cid__cast_info__id tn__2106415 cid__cast_info__id tn__7627157 idx__cast_info__548431 tn__228532 idx__movie_info__107716 tt__lord idx__title__83725 tn__519319 idx__title__83725 tt__n4632 cid__name__name_pcode_nf

...
```

### Embedding Construction
Finally, the last step is to take all these sentences and feed them to the training algorithm, which takes as input the generated walks and trains embeddings using the walks as training corpus.
As configuration parameters, we decided to keep the default ones of *EmbDI*, that is to use *Word2Vec* as training algorithm and, in particular, *Skip-gram* as learning method, with number of dimensions to be used equals to *300* and size of the context window equals to *3*.
The process produced *4.179.184* final embeddings.

```
4179184 300
cid__cast_info__id 0.35783935 0.09026848 0.5204098 -0.30514708 0.3483492 -0.32001132 -0.26266673 -0.06659785 -0.047488194 0.14173894 0.17314543 0.2563694 0.1987069 -0.10807257 -0.06367881 -0.13151994 0.06352358 -0.0068772538 0.06527058 0.15579878 0.017478429 0.037504267 0.05113076 0.24355175 -0.21051376 -0.14102156 -0.0734633 -0.09085068 0.16163963 -0.035228997 0.02496257 -0.18009414 0.26522982 -0.0037418643 0.016840426 -0.2533244 -0.07771756 -0.05089202 0.095280215 0.0730585 -0.06917365 0.010984637 0.05176641 0.15608786 -0.11699749 0.0011402395 -0.035569776 -0.13503733 0.23643132 -0.059274007 0.15535057 -0.26108596 0.115795195 0.05892102 0.4775899 0.12726042 -0.2599606 0.4086665 0.16716976 -0.22544886 0.12326058 0.11276101 -0.048645742 -0.17349063 0.103675604 0.009231482 0.23508286 0.12645042 0.63745826 0.014978187 0.039218485 0.2760019 -0.18508509 0.013079162 -0.26339015 0.07645482 -0.20101348 -0.19416437 0.10791811 0.050859343 0.01048688 -0.0157575 -0.16730537 0.11364455 0.19801323 -0.13440756 -0.28940508 -0.10471786 -0.28117007 0.18172722 0.06680208 0.16629717 0.19242206 -0.10971068 -0.19109724 -0.108425006 0.25900033 0.14031799 0.09817839 0.0004084947 -0.10144412 0.17182952 -0.1275155 0.33373097 0.14283103 0.009269977 -0.019339368 0.24926807 -0.06781546 -0.044509385 -0.10127374 0.034593657 -0.08963413 -0.13039559 -0.16067614 0.096633896 -0.10557844 -0.23638688 -0.12353227 -0.08281243 -0.009466769 0.035744123 0.13025366 0.18360265 0.094200656 0.11584563 0.24417311 -0.25919995 0.044404946 -0.35927707 -0.058690567 -0.007552366 0.4324664 0.024190523 -0.06478646 -0.28637588 -0.029069584 0.19410293 -0.2031938 0.045602366 -0.060319666 0.05389464 0.25285786 0.23044425 0.05022775 0.05789589 -0.017524607 0.31501368 -0.15032312 0.07527011 -0.04184288 0.10922116 -0.15286474 -0.08602699 -0.020160398 -0.015088732 -0.05736754 0.2710062 0.17749926 -0.28276303 -0.061226472 -0.028202629 0.2711332 -0.20575137 -0.186486 -0.13347305 -0.2922679 0.23181196 0.086026005 -0.2302128 0.042264 -0.18333207 0.24346878 -0.030965935 0.11496008 0.21281993 0.16265477 0.059055332 0.0517477 0.40283835 -0.31901675 -0.12134828 0.22315115 0.0037016028 0.10591487 0.20266591 -0.05219141 -0.06517512 -0.05789698 -0.098333 0.14812076 0.06561888 0.08857096 0.28296843 -0.009202234 -0.41281295 0.023726348 -0.16519299 -0.106206715 0.23904245 0.04235076 -0.01878466 0.055920742 0.27051157 0.05869706 0.39527348 0.11394294 -0.17349346 0.10769116 0.39242008 0.1209063 -0.22255556 0.13355808 0.27260628 0.011314887 0.1971161 -0.21484554 0.031988326 0.07115758 -0.28868002 -0.1978277 -0.14618969 0.029123887 -0.19528079 0.2491128 0.01774202 0.15702741 0.013275008 0.06590225 0.12436524 0.16508868 -0.14585918 -0.241522 -0.06493754 -0.0097894585 0.0037239029 -0.045284793 0.117635846 0.039047807 0.06557034 0.11930993 -0.079012394 -0.06834731 -0.3382262 -0.038008783 0.134241 -0.36678505 0.07151187 -0.33530885 -0.009150737 0.16969873 -0.010209005 -0.07330098 0.09804699 -0.023901077 -0.40959457 -0.07764726 -0.011595885 0.14994164 0.12552443 -0.08944777 0.021113671 0.06748307 -0.07430027 0.065108985 0.21960354 -0.30087867 0.0073028 0.17847413 -0.04493163 -0.26870754 -0.0016246935 -0.11380809 0.20331004 0.0825333 0.113387816 0.19042975 0.16259336 0.12733673 -0.09086014 -0.007184363 -0.024417821 0.22920975 0.22618146 0.22739618 0.27466843 -0.07688189 0.2772259 -0.2556025 -0.012012051 0.11896214 -0.17797837 0.2636039 -0.101787984 0.21333335 0.36504355 0.045386337 0.13849057 0.08587369 -0.0059740734
tn__1 0.17930093 -0.09142575 0.13025625 -0.13987246 0.15795128 -0.23172218 0.053887077 -0.28073913 0.06475015 0.10521704 0.0747659 0.16771579 0.29155105 -0.22669818 -0.054823503 -0.19611233 -0.22108477 0.18897401 0.1387693 0.03233738 0.11311568 -0.0924699 -0.32668045 0.2636749 -0.47781122 -0.056094717 0.042853132 0.047652744 0.35801494 -0.099138856 -0.19909813 -0.097361445 -0.29900596 0.35237077 -0.34305516 0.008370778 0.27087796 0.09974467 0.07845239 0.10826513 -0.028940767 0.04195136 -0.0591843 -0.04166096 -0.32250586 -0.016169108 -0.022530757 0.037331842 0.17512442 -0.12597819 0.0023031866 -0.17123225 -0.034606025 -0.08749069 0.3237761 -0.060568094 0.11323135 -0.07064527 0.4523828 -0.1566417 -0.25307173 0.2595879 -0.29397774 0.43714982 -0.03494516 0.26596323 0.0047569843 -0.1515291 0.3740311 0.06487778 -0.017157877 0.07162671 -0.13421547 -0.32384714 -0.10345116 -0.30945957 -0.29643485 -0.20464864 -0.11366968 0.088446826 -0.016362842 0.019439204 0.18144178 -0.038216356 0.031047681 -0.04183735 -0.09340055 0.032342713 -0.3774005 0.23395337 -0.2784426 -0.14665742 0.17795993 -0.28651002 -0.15411648 -0.31794894 -0.4019272 0.050374363 0.11017604 0.12729563 -0.34940302 0.31603482 0.22175068 -0.03901575 0.1954573 0.27540898 0.072367735 0.2795549 -0.09354923 -0.11838083 -0.10018332 0.26731408 -0.18620145 -0.010052366 0.0059936456 0.06727401 0.184651 -0.033302564 -0.16792628 -0.42666355 0.05205271 0.047659956 0.017617807 -0.1476152 0.2792051 -0.30337268 0.059940726 -0.0147954915 -0.069111384 -0.10498007 -0.17775713 -0.0628631 0.24178118 0.055094317 0.41378802 -0.048622046 -0.091939606 0.10144112 0.0021544371 0.13486925 0.03501151 0.14267476 0.17703065 0.16700597 -0.033935588 0.06455656 -0.2010721 0.096174136 -0.19621713 0.18712898 0.011521017 0.022833379 -0.050118472 -0.007093148 -0.08177355 -0.22442445 -0.11581571 0.26946232 0.07452359 -0.12592737 0.19221386 0.36057812 -0.051634785 -0.14111428 0.017142938 0.15158883 0.12991408 -0.27645403 0.113947995 -0.09103817 0.07833296 0.1681714 -0.117377296 0.0066901506 0.062881455 0.06829864 -0.020707464 0.14222533 -0.012417379 -0.0455827 -0.15180431 -0.020623734 0.23230687 0.07575161 0.11442765 -0.16251537 -0.0488991 0.13148038 0.17140085 0.14503437 -0.010569984 0.1229868 0.015938995 0.27323645 0.14037846 -0.26318297 0.024543922 -0.2114747 -0.44544452 -0.036364093 -0.012643718 -0.11942415 0.0993801 0.118459255 -0.05164453 0.06666055 0.037584044 -0.16587445 0.07603128 0.04791906 0.033848144 -0.11502371 -0.01518633 0.21127464 -0.27952886 0.14535257 -0.12439029 -0.08706023 -0.010871644 -0.12601829 0.10787277 -0.14918047 0.121450394 0.04658887 0.19712257 -0.14831127 -0.047169756 0.08925967 0.28329802 0.068087466 -0.054900505 0.15434219 -0.12299036 -0.094191246 -0.12444215 -0.0071410956 0.05552283 0.14161272 0.09810762 0.23003635 -0.1033792 -0.1807974 -0.1695115 -0.053956106 0.060831197 -0.15710646 -0.1094411 -0.018956749 0.21490768 0.099752046 -0.22338058 0.05147894 -0.13903715 0.10538395 0.38468105 -0.09417374 -0.17987245 -0.011408303 0.14722276 0.06209728 0.049197808 -0.04890676 -0.056724127 -0.14953007 -0.2326627 0.099229686 -0.14824525 -0.17595962 0.005600773 -0.2836629 -0.03329623 -0.15145952 -0.2321835 -0.056634806 0.25353664 -0.2884102 0.13172096 0.35686544 -0.118369766 -0.1035686 0.13072129 -0.061630655 0.061684884 0.35776126 -0.0078038154 0.22698876 0.1429465 0.07875978 -0.09794773 0.07683084 -0.039998535 -0.24161035 -0.037780557 -0.021591913 0.57715774 0.18525587 0.092136584 -4.867828e-05 0.089967124 0.079391934
cid__name__name -0.039210204 -0.06335678 -0.22404583 -0.13330282 -0.30212158 -0.3552669 -0.30727163 -0.41234168 -0.20647372 -0.0202369 0.0610646 -0.30303186 -0.44002894 0.2530534 -0.47174478 0.26706278 0.16418058 -0.48096827 0.33262613 -0.030877212 0.38249367 -0.47308347 -0.059908696 -0.3000688 -0.16902053 0.070702836 -0.11271938 0.4714423 0.0055647176 0.038128972 -0.13758026 -0.12496759 0.3586617 -0.059510004 -0.19602996 -0.32557535 -0.029406799 0.06320658 0.18132788 -0.08145399 -0.34670413 0.32318488 -0.13099419 0.030318907 0.14612766 -0.11192 0.020578718 -0.34098417 0.004990968 -0.18570833 0.09781367 0.11333143 0.057519466 -0.028001541 0.29060474 -0.059507944 0.3963247 0.0050053773 -0.112317935 -0.08626351 0.14544915 0.015799293 0.19109982 0.087949246 -0.15127613 -0.21073592 0.017893583 0.33244264 -0.023480993 0.35085556 -0.1191288 0.2004401 -0.3842513 0.087675646 -0.1985592 -0.24545649 0.35280198 -0.1279247 -0.10705803 0.18106212 0.1468469 -0.15748453 0.08990896 -0.119961195 0.40371242 0.16175434 -0.29465306 0.061258536 -0.22680853 0.29367077 -0.23556732 0.19247095 0.01817373 -0.047883593 0.2798201 0.12213378 -0.06465709 0.59402376 0.025928685 -0.3035532 0.3368378 -0.010984672 0.12110508 0.3040442 -0.31047192 -0.07571356 -0.1855763 -0.2069577 -0.17154546 0.2738722 0.14072758 0.088001125 -0.1376548 -0.29239023 -0.23020416 0.43801665 0.2282878 0.016301319 0.02822925 -0.076436944 -0.14805631 -0.14274313 -0.14128755 0.12440997 0.3309525 -0.17999408 0.090983756 -0.105743 0.19939011 -0.038505565 -0.09693426 -0.13481344 0.18083021 -0.2881981 0.3728363 -0.1661691 0.14102899 -0.022713779 -0.047397435 0.42889994 -0.008289725 -0.07335485 0.0979838 -0.096067846 0.109586895 0.014258688 0.12315766 -0.27442518 -0.3467684 0.009709708 -0.2699284 0.1710712 0.03952216 0.073092945 0.07222713 -0.012925957 -0.21530403 0.11031981 -0.07760873 -0.11141604 0.11403901 0.5372587 -0.4473179 0.062628984 -0.03431618 0.14478567 0.045808833 -0.20963055 -0.17792697 -0.2121505 -0.1650009 -0.06990899 0.0865645 -0.23676871 -0.15258546 -0.22559088 0.5269771 -0.019727591 0.12695684 0.105280966 -0.001770209 0.11838115 -0.041735236 0.26152855 -0.05383577 -0.18291543 -0.0645422 -0.48867613 -0.13520868 0.090164885 0.1969908 -0.2167148 -0.26693738 -0.15454555 0.34678292 -0.25057504 0.15422963 -0.4439602 0.19691615 -0.004033277 0.28896922 -0.10201062 0.15039277 -0.41911635 0.046938315 0.052986547 0.44115594 0.07416627 0.10841036 0.36444607 0.30909678 0.10233357 0.4516681 0.3319006 0.017229898 -0.0825026 -0.43139273 0.116346166 -0.12407608 0.029440928 -0.020621724 0.113841526 0.00043974764 -0.23484492 0.24994734 0.044876643 0.037980992 0.06025046 -0.0011745495 -0.30938914 0.05166501 -0.09550459 0.0044303644 -0.065369315 -0.14435908 0.16879244 -0.027163362 0.1809353 -0.12693028 0.0024959054 0.06415364 -0.3793468 -0.32725102 -0.02876527 0.22952996 -0.37727174 0.20649906 -0.21038955 -0.17638728 -0.25564814 -0.54752785 0.030306106 -0.18280946 0.016336216 -0.04525612 -0.34042728 -0.10422381 0.3402074 -0.07227341 0.034814447 0.015611385 0.068010665 -0.20254157 0.24439833 0.23418014 -0.10019105 -0.03913135 -0.19921342 0.3500311 -0.29608595 0.4839862 0.20324771 -0.109507844 0.047284182 0.21984033 -0.17789054 0.08482719 0.19733581 -0.24359426 -0.10568514 -0.30821005 -0.09216543 0.34704742 -0.09334695 0.01948454 0.030633062 0.093475685 0.12892754 -0.30034903 -0.078608125 -0.13403134 0.044395205 0.2594294 0.37502843 0.019583313 0.27737603 0.28465876 -0.21885629 0.030947357 -0.10413909
...
```

*EmbDI* is able to generate very effective embeddings, since it takes into
account both single numeric tokens (*tn__*), and single categorical tokens (*tt__*), as well as column attributes (*cid__*), and record indices (*idx__*).

## RETRO Wrapper
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