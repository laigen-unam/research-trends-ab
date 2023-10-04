# Unsupervised learning and natural language processing point out bias in research trends of a superbug 
## Motivation
Antibiotic-resistance Acinetobacter baumannii is a very important nosocomial pathogen worldwide. Thousands of studies have been conducted about this pathogen. However, there has not been any attempt to use all this information to highlight the research trends concerning this pathogen. Here we use unsupervised learning and natural language processing (NLP), two areas of Artificial Intelligence, to analyse the most extensive database of articles created (5500+ articles, from 851 different journals, published over 3 decades). 

## Supplementary data
* Description of supplementary tables: supplementary_data/Supplementary_Data.pdf
* Supplementary Table S1, clustering table: supplementary_data/Table_S1.xlsx.
* Supplementary Table S2, cluster labeling table: supplementary_data/Table_S2.xlsx. 
* Supplementary Table S3, table with 10% of clusters with manually assigned labels: supplementary_data/Table_S3.xlsx.

## Input data
Files with PMID, title, abstract, publication date, authors and journal ISO
abbreviation from PubMed system.
* Up to May 2022 Downloaded: input_data/abs.txt
* Up to May 2022 Lemmatized: input_data/abs_lemmatized.txt
* From May 2022 to May 2023 Downloaded: input_data/abs-2022-2023.txt 
* From May 2022 to May 2023 Lemmatized: input_data/abs-2022-2023_lemmatized.txt 

##  Scripts
* scripts/abstrsct_ext.py

Download PMID, title, abstract, publication date, authors and journal ISO
abbreviation from PubMed system.

* scripts/lemmas_with_stanza.py

Title and abstract lemmatization with Stanza.

* scripts/tf-idf.py

Transformation of each publication into a numeric vector using the tf-idf weighting scheme.

* scripts/svd_pca.py

Dimensionality reduction of the tf-idf matrix using a truncated singular value decomposition method (truncated-SVD).

* scripts/silhouette.py

Quality of different clustering analyses with values of k from 50 to 500 using the Silhouette coefficient.

* scripts/clustering.py

Clustering with k = 113.

* scripts/clustering_prediction.py

Cluster prediction for new publications.

