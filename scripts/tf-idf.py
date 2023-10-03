import os
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.corpus import stopwords
from joblib import dump, load
from scipy.sparse import csr_matrix
from gensim.parsing.preprocessing import strip_numeric, strip_short


def tfidf(Abstract, outputPath):
    t0 = time()

    print("\nReading input file...")
    dic_abs = []
    with open(Abstract, mode='r') as file:
        for line in file:
            line = line.strip("\n\r")
            # CFMC: To append title to abstract
            list_line = line.split("\t")
            line = list_line[1] + " " + list_line[2]
            # CFMC: Ignore words less than 3 characters and numbers
            line = strip_numeric(line)
            line = strip_short(line)
            dic_abs.append(line)
    print("Done!\n")
    print(f"{len(dic_abs)} abstracts read!")

    print("Vectorizing input file...")
    pf = stopwords.words('english')
    # CFMC: We ignore terms in more than 50% (0.5) of documents and with frequency less than 2
    vectorizer = TfidfVectorizer(stop_words=pf, max_df = 0.5, min_df=2)
    X = vectorizer.fit_transform(dic_abs).toarray()
    X = csr_matrix(X, dtype='double')
    names = vectorizer.get_feature_names()
    print("Found features:", len(names))
    nc, i_features = X.get_shape()
    print(f"Articles: {nc}")
    print("Features: ", i_features)
    print("Done!\n")

    print("Writing output files...")
    with open(os.path.join(outputPath, "clases_Abstracts.txt"), mode="w") as oFile:
        oFile.write(str(names))
        oFile.write('\n')

    dump(X, os.path.join(outputPath, "Abstracts_vect.joblib"))
    # CFMC: Save vectorizer
    dump(vectorizer, os.path.join(outputPath, "vectorizer.joblib"))

    print("Done!\n")

def tfidf_add_publications(abstracts_File, tfidf_File, output_Path):
    t0 = time()

    print("\nReading input file...")
    dic_abs = []
    with open(abstracts_File, mode='r') as file:
        for line in file:
            line = line.strip("\n\r")
            # CFMC: To append title to abstract
            list_line = line.split("\t")
            line = list_line[1] + " " + list_line[2]
            # CFMC: Ignore words less than 3 characters and numbers
            line = strip_numeric(line)
            line = strip_short(line)
            dic_abs.append(line)
    print("Done!\n")
    print(f"{len(dic_abs)} abstracts read!")

    print("Loading tf-idf file of clusterized collection...")
    vectorizer = load(tfidf_File)
    print("Vectorizing input file...")
    # CFMC: We ignore terms in more than 50% (0.5) of documents and with frequency less than 2
    X = vectorizer.transform(dic_abs).toarray()
    X = csr_matrix(X, dtype='double')
    names = vectorizer.get_feature_names()
    print("Found features:", len(names))
    nc, i_features = X.get_shape()
    print(f"Articles: {nc}")
    print("Features: ", i_features)
    print("Done!\n")

    print("Writing output files...")
    with open(os.path.join(output_Path, "clases_Abstracts.txt"), mode="w") as oFile:
        oFile.write(str(names))
        oFile.write('\n')

    dump(X, os.path.join(output_Path, "Abstracts_vect.joblib"))
    print("Add publications to tf-idf done! ...\n")

# Publications for clustering
input_file = "abs_lemmatized.txt"
output_path = "/temp/updating-2022-2023"
tfidf_path = "/temp/vectorizer.joblib"
tfidf_add_publications(input_file, tfidf_path, output_path)
