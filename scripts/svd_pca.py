from sklearn.decomposition import TruncatedSVD
import sys
import os
from pathlib import Path
import datetime
import time
from sklearn.decomposition import PCA
from joblib import load
from joblib import dump
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

## Object to record log (unused in version with reticulate in R)
class Logger(object):
        "this class is for print the output of the script to both stdout and log file"
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(os.path.join(Path(outputLog).parent.absolute(), 'svd_log.txt'), "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass

# CFMC: To load selectf_result object for ShinyApp
def load_selectf_result(outresult):
    selectf_result = load(outresult)
    return selectf_result

def reduc(file_matrix, output, outsvd, outresult, method):

    start_time = time.time()
    print("\n\n**********************************************************************\n")
    print("Dimensionality reduction\n")
    print("Run started at: ", datetime.datetime.now(), '\n')

    print("Saving supplementary_data in file: ", output)

    print("\nReading input files...")
    X = load(file_matrix)
    y = []

    print("Done!\n")
    nc, i_features = X.get_shape()
    print(f"Articles: {nc}")
    print("Features: ", i_features)

    if method == "SVD":
        print("Performing SVD reduction...")
        nc = 300
        svd = TruncatedSVD(n_components=nc, random_state=42, n_iter=10)
        # CFMC: this page reccomend normalization for document clustering: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X_reduced = lsa.fit_transform(X)
        exp_var = svd.explained_variance_ratio_.cumsum()

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        params = svd.get_params()
    if method == "PCA":
        print("Performing PCA reduction...")
        pca = PCA(n_components=nc, random_state=42)
        X_reduced = pca.fit_transform(X.toarray())
        exp_var = pca.explained_variance_ratio_.cumsum()


    cont = 1
    for i in exp_var:
        if(i >= 0.7):
            break
        cont += 1
    print("Done!\n")

    f_abs, f_features = X_reduced.shape
    print(f"TF-IDF matrix reduced to dimensions: {f_features}")
    dump(X_reduced, output)
    # CFMC: Save svd
    dump(svd, outsvd)

    print("Total time: %s seconds." % (time.time() - start_time))
    print("\n**********************************************************************\n")

    # CFMC: To save result before return
    selectf_result = [str(time.time() - start_time), str(i_features), str(f_features)]
    dump(selectf_result, outresult)
    return selectf_result

def reduc_add_publications(file_matrix, output, insvd, outresult, method):
    start_time = time.time()
    print("\n\n**********************************************************************\n")
    print("Add new publications to dimensionality reduction\n")
    print("Run started at: ", datetime.datetime.now(), '\n')
    print("Saving supplementary_data in file: ", output)
    print("\nReading input files...")
    X = load(file_matrix)
    print("Done!\n")
    nc, i_features = X.get_shape()
    print(f"Articles: {nc}")
    print("Features: ", i_features)
    if method == "SVD":
        print("Performing SVD reduction...")
        print("Loading SVD file of clusterized collection...")
        svd = load(insvd)
        # CFMC: this page reccomend normalization for document clustering: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X_reduced = lsa.transform(X)
        exp_var = svd.explained_variance_ratio_.cumsum()

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    print("Done!\n")
    f_abs, f_features = X_reduced.shape
    print(f"TF-IDF matrix reduced to dimensions: {f_features}")
    dump(X_reduced, output)
    print("Total time: %s seconds." % (time.time() - start_time))
    print("\n**********************************************************************\n")
    # CFMC: To save result before return
    selectf_result = [str(time.time() - start_time), str(i_features), str(f_features)]
    dump(selectf_result, outresult)
    return selectf_result

# Publications for clustering
input = "Abstracts_vect.joblib"
output = "Abstracts_vect_reduced.joblib"
input_svd = "svd.joblib"
output_result = "selectf_result.joblib"
reduc_add_publications(input, output, input_svd, output_result, "SVD")
