from math import ceil
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from concurrent.futures import ProcessPoolExecutor
import time
import matplotlib.pyplot as plt
import datetime
import os
from joblib import load


def testK(k):
    labels = KMeans(n_clusters=k, init="k-means++", random_state=200).fit(tf_matrix).labels_
    score = silhouette_score(tf_matrix, labels, metric="euclidean", sample_size=1000, random_state=200)
    return score

def load_global_scores(plots):
    g_scrs = load(os.path.join(plots, 'scores.joblib'))
    return g_scrs

def eval_sil(searchm, lim_inf, lim_sup, file_matrix, plots, threads, iter=10, step=1):
    if not os.path.exists(plots): os.makedirs(plots)
    start_time = time.time()

    print("\n\n**********************************************************************\n")

    print("Run started at: ", datetime.datetime.now(), '\n')

    print("Running with input parameters:\n")
    print("Search mode: ", searchm)
    print("Lower limit: ", lim_inf)
    print("Upper limit: ", lim_sup)
    if searchm == "l":
        print("Step: ", step)
    elif searchm == "b":
        print("Max iter: ", iter)
    print("Vect matrix: ", file_matrix)
    print("Joblib dict output: ", os.path.join(plots, 'scores.joblib'))
    print(f"Running in {threads} threads...")

    print("\nReading matrix input...")
    global tf_matrix
    tf_matrix = load(file_matrix)
    print("Done!")

    print(f"Searching best K(clusters) for {tf_matrix.shape[0]} articles with {tf_matrix.shape[1]} features...")
    global global_scores
    global_scores = {}
    threads = int(threads)
    if searchm == "Linear":
        K = range(lim_inf, lim_sup + step, step)
        print(f"\nSearching in the interval between {K[0]} and {K[-1]}, with {len(K)} iterations...")
        with ProcessPoolExecutor(max_workers=threads) as executor:
            scores = list(executor.map(testK, K))
        for i in range(len(scores)):
            print("Silhouette score for k(clusters) = " + str(K[i]) + " is " + str(scores[i]))
        max_k = K[max(range(len(scores)), key=scores.__getitem__)]
        print(f"\nMax average score found in k = {max_k}")
        for k, score in zip(K, scores): global_scores[k] = score
    elif searchm == "Binary":
        search = True
        while (search):
            step = ceil((lim_sup - lim_inf) / iter)
            K = range(lim_inf, lim_sup + step, step)
            print(
                f"\nSearching in the interval between {K[0]} and {K[-1]}, with {len(K)} iterations and step={step}...")
            with ProcessPoolExecutor(max_workers=threads) as executor:
                scores = list(executor.map(testK, K))
            for i in range(len(scores)):
                print("Silhouette score for k(clusters) = " + str(K[i]) + " is " + str(scores[i]))
            for k, score in zip(K, scores): global_scores[k] = score
            if step > 1:
                max_k = K[max(range(len(scores)), key=scores.__getitem__)]
                print(f"\nMax average score found in k = {max_k}")
                lim_inf = max_k - ceil(step / 2)
                if lim_inf <= 1: lim_inf = 2
                lim_sup = max_k + ceil(step / 2)
                if (lim_inf <= 0): lim_inf = lim_inf - lim_inf + 2
            else:
                search = False
    else:
        raise ("Select a valid search mode.")
    global_scores = dict(sorted(global_scores.items(), key=lambda item: item[1], reverse=True))
    print(f"\n\nMax silhouette average score found in k = {list(global_scores.keys())[0]}\n")
    joblib.dump(global_scores, os.path.join(plots, 'scores.joblib'))
    print("Calculated in %s seconds." % (time.time() - start_time), '\n')
    print("Plotting...")
    global_scores1 = dict(sorted(global_scores.items(), key=lambda item: item[0], reverse=True))
    plt.plot(list(global_scores1.keys()), list(global_scores1.values()))
    plt.title("Average silhouette scores of k(clusters) tested")
    plt.xlabel("k(number of clusters)")
    plt.ylabel("Average silhouette score")
    plt.savefig(os.path.join(plots, 'scores.png'))
    print("Total time: %s seconds." % (time.time() - start_time), '\n')
    print("\n**********************************************************************\n")

    # CFMC: to shutdown the ProcessPoolExecutor
    executor.shutdown()

    return global_scores

g_scores = eval_sil("Linear", 50, 500,
                    file_matrix="Abstracts_vect_reduced.joblib",
                    plots="supplementary_data/",
                    threads=40
                    )
