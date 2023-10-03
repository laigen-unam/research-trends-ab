import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib
from joblib import load
from joblib import dump

def MatplotlibClearMemory():
    allfignums = matplotlib.pyplot.get_fignums()
    for i in allfignums:
        fig = matplotlib.pyplot.figure(i)
        fig.clear()
        matplotlib.pyplot.close( fig )

# CFMC: We add parameters clusterer and cluster_labels to use clustering done in function clustering()
def plotK(n_clusters, plots, p, clusterer, cluster_labels):
    matplotlib.use("Agg", force = True)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(10, 20)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, tf_matrix.shape[0] + (int(n_clusters) + 1) * 10])

    # CFMC: We use clustering done in function clustering()
    # CFMC: We use lables from clustering done in function clustering()
    silhouette_avg = global_scores[int(n_clusters)]
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_reduced, cluster_labels)
    y_lower = 10

    for i in range(int(n_clusters)):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / int(n_clusters))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / int(n_clusters))
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(
        f"Silhouette analysis for KMeans clustering with k = {n_clusters} and p = {p}",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(os.path.join(plots, f"k{int(float(n_clusters))}_perp{int(float(p))}.png"))
    plt.figure().clear()
    plt.close("all")
    plt.cla()
    plt.clf()
    # MatplotlibClearMemory()

def clustering(Abstract_reduc, Abstract, outputpath, k, perplexity, scores, svd_file, vec_file, kmeans_file):

    global global_scores
    global_scores = scores
    # print(global_scores)

    k = str(k)
    perplexity = str(perplexity)
    filename = "k"+k+"_perp"+str(int(float(perplexity)))+".txt"
    # CFMC: To save cluster label with top terms per cluster
    filename_clstr = "k" + k + "_perp" + str(int(float(perplexity))) + "_clst.txt"
    filename_job = "k" + k + "_perp" + str(int(float(perplexity))) + "_clst.joblib"

    print("\nReading input files...")
    global X_reduced
    X_reduced = load(Abstract_reduc)

    global tf_matrix
    tf_matrix = load(Abstract)

    print("Done!\n")

    #Clusterizacion
    print("Perfoming k-means clusterization with {} clusters...".format(k))
    # class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
    # Code from silhouette.py: labels = KMeans(n_clusters=k, init="k-means++", random_state=200).fit(tf_matrix).labels_
    # CFMC: we change random_state to use the same random_state in silhouette evaluation, we remove n_jobs=-1 as it is deprecated.
    # Also, we separate fit and predict to obtain top terms per cluster (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)
    kmeans = KMeans(n_clusters=int(k), init="k-means++", random_state=200).fit(X_reduced)
    # Save the kmeans clustering model
    dump(kmeans, os.path.join(outputpath, kmeans_file))
    y_pred = kmeans.labels_
    # CFMC: To obtain top terms per cluster (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html)
    svd = load(svd_file)
    original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vectorizer = load(vec_file)
    terms = vectorizer.get_feature_names()
    # CFMC: Top terms per cluster
    print("Found features:", len(terms))
    nc, i_features = tf_matrix.get_shape()
    print(f"Articles: {nc}")
    print("Features: ", i_features)
    print("Done!\n")
    hash_top_terms = {}
    for i in range(int(k)):
        for ind in order_centroids[i, :10]:
            if i in hash_top_terms:
                hash_top_terms[i].append(terms[ind])
            else:
                hash_top_terms[i] = [terms[ind]]
    print("Done!\n")

    #t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(verbose=1, perplexity=int(float(perplexity)), random_state=42)
    X_embedded = tsne.fit_transform(X_reduced)
    print("Done!\n")
    tf_matrix = X_embedded

    global X
    X = tf_matrix

    print("Saving output data files...")
    with open(os.path.join(outputpath, filename), mode='w', encoding='utf8') as oFile:
        oFile.write("dim1\tdim2\tCluster\n")
        for i in range(len(y_pred)):
            oFile.write("{}\t{}\t{}\n".format(X_embedded[i,0], X_embedded[i,1], y_pred[i]))

    with open(os.path.join(outputpath, filename_clstr), mode='w', encoding='utf8') as oFile:
        oFile.write("Cluster\ttop_terms\n")
        for c, v in hash_top_terms.items():
            top_terms_per_cluster = " ".join(v)
            oFile.write("{}\t{}\n".format(c, top_terms_per_cluster))

    dump(hash_top_terms, os.path.join(outputpath, filename_job))

    print("Plotting preview...")

    plotK(k, outputpath, perplexity, kmeans, y_pred)

    print("Done! Shutting down.\n")

# CFMC: To calculate and save silhouette scores for clusters
def save_silhouette_scores(n_clusters, path_output, path_kmeans, Abstract_reduc):
    df_silhouette_scores = pd.DataFrame(
        columns=['Cluster id', 'silhouette_score', 'cluster_size'])
    kmeans = load(path_kmeans)
    cluster_labels = kmeans.labels_
    X_reduced = load(Abstract_reduc)
    sample_silhouette_values = silhouette_samples(X_reduced, cluster_labels)
    for i in range(int(n_clusters)):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        silhouette_avg = sum(ith_cluster_silhouette_values) / size_cluster_i
        new_row = {'Cluster id': int(i),
            'silhouette_score': silhouette_avg,
            'cluster_size': size_cluster_i}
        df_silhouette_scores = df_silhouette_scores.append(new_row, ignore_index=True)

    df_silhouette_scores.to_csv(os.path.join(path_output, "silhouette_values_clusters.tsv"), sep='\t', index=False, header=True)

path = "/project-ab/"
scrs = load(os.path.join(path, "supplementary_data/scores.joblib"))
clustering(os.path.join(path, "temp/Abstracts_vect_reduced.joblib"),
           os.path.join(path, "temp/Abstracts_vect.joblib"),
           os.path.join(path, "supplementary_data/clusters"),
           113, 30,
           scrs,
           os.path.join(path, "temp/svd.joblib"),
           os.path.join(path, "temp/vectorizer.joblib"),
           os.path.join(path, "kmeans_113.joblib")
           )
