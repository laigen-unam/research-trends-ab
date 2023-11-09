# Unsupervised learning and natural language processing point out bias in research trends of a superbug 
## Motivation
Antibiotic-resistance Acinetobacter baumannii is a very important nosocomial pathogen worldwide. Thousands of studies have been conducted about this pathogen. However, there has not been any attempt to use all this information to highlight the research trends concerning this pathogen. Here we use unsupervised learning and natural language processing (NLP), two areas of Artificial Intelligence, to analyse the most extensive database of articles created (5500+ articles, from 851 different journals, published over 3 decades). 

## Supplementary data
1. Description of supplementary methods, figures and tables.
```shell
supplementary_data/Frontiers_Supplementary_Material.pdf
```
2. Supplementary Table S1, clustering table.
```shell
supplementary_data/Supplementary_Table_S1.xlsx
```
3. Supplementary Table S2, cluster labeling table.
```shell
supplementary_data/Supplementary_Table_S2.xlsx. 
```
4. Supplementary Table S3, table with 10% of clusters with manually assigned labels.
```shell
supplementary_data/Supplementary_Table_S3.xlsx.
```
5. Supplementary Table S4, table with 644 recovered from May 13th, 2022 to May 23rd, 2023 
   assigned to clusters by the prediction of the model.
```shell
supplementary_data/Supplementary_Table_S4.docx.
```
6. Supplementary Figure S1. Total of publications per journal. 
   We only show journals with 10 or more publications.
```shell
supplementary_data/Supplementary_Figure_S1.png.
```
7. Supplementary Figure S2. Total of publications per year. 
   The year 2022 included publications until May 12th.
```shell
supplementary_data/Supplementary_Figure_S2.png.
```
8. Supplementary Figure S3. Average of LDA terms, terms from centroids (Centroid terms), 
   and overlapped terms between both (Overlapped terms) appearing in the manually 
   assigned short phrase (label) of a 10% percent of randomly selected clusters: 
   5, 13, 22, 24, 43, 49, 77, 78, 81, 87, 91, 109.
```shell
supplementary_data/Supplementary_Figure_S3.png.
```
9. Supplementary Figure S4. Distribution of publications assigned to clusters 
by means of the prediction of the trained model.
```shell
supplementary_data/Supplementary_Figure_S4.png.
```
## Input data
Files with PMID, title, abstract, publication date, authors and journal ISO
abbreviation from PubMed system.
1. Up to May 2022 Downloaded: 
```shell
input_data/abs.txt
```
2. Up to May 2022 Lemmatized: 
```shell
input_data/abs_lemmatized.txt
```
3. From May 2022 to May 2023 Downloaded: 
```shell
input_data/abs-2022-2023.txt 
```
4. From May 2022 to May 2023 Lemmatized: 
```shell
input_data/abs-2022-2023_lemmatized.txt 
```
