# Research trends of _Acinetobacter baumannii_ 

Unsupervised learning and natural language processing point out bias in research trends of Acinetobacter baumannii

## Supplementary data
* Description of supplementary tables: Supplementary_Data.pdf
* Supplementary Table S1, clustering table: Table_S1.xlsx.
* Supplementary Table S2, cluster labeling table: Table_S2.xlsx. 
* Supplementary Table S3, table with 10% of clusters with manually assigned labels: Table_S3.xlsx.

## Input data
Files with PMID, title, abstract, publication date, authors and journal ISO
abbreviation from PubMed system.
* Up to May 2022 Downloaded: abs.txt
* Up to May 2022 Lemmatized: abs_lemmatized.txt
* From May 2022 to May 2023 Downloaded: abs-2022-2023.txt 
* From May 2022 to May 2023 Lemmatized: abs-2022-2023_lemmatized.txt 

##  Scripts
* abstrsct_ext.py

Download PMID, title, abstract, publication date, authors and journal ISO
abbreviation from PubMed system.

* lemmas_with_stanza.py

Title and abstract lemmatization with Stanza.

* tf-idf.py

Transformation of each publication into a numeric vector using the tf-idf weighting scheme.

* svd_pca.py

Dimensionality reduction of the tf-idf matrix using a truncated singular value decomposition method (truncated-SVD).

* silhouette.py

Quality of different clustering analyses with values of k from 50 to 500 using the Silhouette coefficient.

* clustering.py

Clustering with k = 113.

* clustering_prediction.py

Cluster prediction for new publications.

