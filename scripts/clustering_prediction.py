import os
from sklearn.manifold import TSNE
from joblib import load

def clustering_prediction(Abstract_reduc, Abstract, outputpath, k, perplexity, kmeans_file):
    print("Abstract_reduc: {}".format(Abstract_reduc))
    print("Abstract: {}".format(Abstract))
    print("outputpath: {}".format(outputpath))
    print("kmeans_file: {}".format(kmeans_file))
    k = str(k)
    perplexity = str(perplexity)
    filename = "k" + k + "_perp" + str(int(float(perplexity))) + "_prediction.txt"
    # CFMC: To save cluster label with top terms per cluster
    print("\nReading input files...")
    global X_reduced
    X_reduced = load(Abstract_reduc)

    global tf_matrix
    tf_matrix = load(Abstract)

    print("Done!\n")

    #Clusterizacion
    print("Perfoming k-means prediction with {} clusters...".format(k))
    # Load the kmeans clustering model
    kmeans = load(kmeans_file)
    # Predict cluster for new publications
    y_pred = kmeans.predict(X_reduced)
    #t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(verbose=1, perplexity=int(float(perplexity)), random_state=42)
    X_embedded = tsne.fit_transform(X_reduced)
    print("Done!\n")

    tf_matrix = X_embedded
    print("Saving output data files...")
    with open(os.path.join(outputpath, filename), mode='w', encoding='utf8') as oFile:
        oFile.write("dim1\tdim2\tCluster\n")
        for i in range(len(y_pred)):
            oFile.write("{}\t{}\t{}\n".format(X_embedded[i,0], X_embedded[i,1], y_pred[i]))

    print("Done!\n")

path = "/proyecto-ab"
clustering_prediction(os.path.join(path, "temp/updating-2022-2023/Abstracts_vect_reduced.joblib"),
           os.path.join(path, "temp/updating-2022-2023/Abstracts_vect.joblib"),
           os.path.join(path, "supplementary_data/clusters/updating-2022-2023"),
           113, 30,
           os.path.join(path, "supplementary_data/clusters/updating-2022-2023/kmeans_113.joblib")
           )
