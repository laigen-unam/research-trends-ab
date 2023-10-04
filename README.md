# Unsupervised learning and natural language processing point out bias in research trends of a superbug 
## ABSTRACT 
### Motivation
Antibiotic-resistance Acinetobacter baumannii is a very important nosocomial pathogen worldwide. Thousands of studies have been conducted about this pathogen. However, there has not been any attempt to use all this information to highlight the research trends concerning this pathogen. Here we use unsupervised learning and natural language processing (NLP), two areas of Artificial Intelligence, to analyse the most extensive database of articles created (5500+ articles, from 851 different journals, published over 3 decades). 
### Results
Clustering k-means found 113 theme clusters, which were defined with representative terms automatically obtained with topic modelling, summarising different research areas. The biggest clusters, all with over 100 articles, are biased toward multidrug resistance, carbapenem resistance, clinical treatment, and nosocomial infections. However, we also found that some research areas, such as ecology, have received very little attention. This approach allowed us to study research themes over time unveiling those of recent interest, such as the use of cefiderocol (a recently approved antibiotic) against A. baumannii . Our results show that unsupervised learning, NLP and topic modelling can be used to describe and analyse the research themes for particular infectious diseases. This strategy should be very useful to analyse other ESKAPE pathogens.

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

