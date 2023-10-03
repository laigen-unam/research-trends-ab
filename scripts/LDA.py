from joblib import dump
from math import floor, ceil
import os
from time import time
from nltk.corpus import stopwords
import gensim
from gensim import models
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, strip_short
from re import sub, findall
# CFMC: Mallet Python wrapper (https://github.com/maria-antoniak/little-mallet-wrapper/blob/master/demo.ipynb)
# pip install little_mallet_wrapper
import little_mallet_wrapper as lmw

# CFMC: Function to round number of word per topic (https://realpython.com/python-rounding/)
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return floor(n*multiplier + 0.5) / multiplier

# CFMC: Function to format Gensim topic output by selecting 10 words by topic. Ignore duplicates
def get_topics_gensim(input_topics, num_t, num_w):
    list_topics = []
    # num_words = round_half_up(num_w/num_t)ceil
    num_words = ceil(num_w/num_t)
    # print("num_words: {}".format(num_words))
    for topic in input_topics:
        num = topic[0]
        word_line = topic[1]
        words = findall(r'"(\w+)"', word_line)
        w = 0
        for word in words:
            if word not in list_topics:
                list_topics.append(word)
                w += 1
                if w == num_words:
                    break
    return " ".join(list_topics)

# CFMC: Function to format Mallet topic output by selecting 10 words by topic. Ignore duplicates
def get_topics_mallet(input_topics, num_t, num_w):
    list_topics = []
    num_words = round_half_up(num_w/num_t)
    # print("num_words: {}".format(num_words))
    for topic, topics in enumerate(input_topics):
        words = findall(r'(\w+)', ' '.join(topics[:15]))
        w = 0
        for word in words:
            if word not in list_topics:
                list_topics.append(word)
                w += 1
                if w == num_words:
                    break
    return " ".join(list_topics)

# CFMC: Function to get Mallet coherence from XML file
def get_mallet_coherence(xml_path, num_t):
    # <topic id='0' tokens='4164.0000' document_entropy='3.3567' word-length='6.9000' coherence='-129.4790'
    with open(xml_path) as xmlFile:
        xml_text = xmlFile.read()
    matches = findall(r"<topic id='([0-9]+)'[^>]+coherence='(-[0-9]+\.[0-9]+)'", xml_text)
    # CFMC: Average topic coherence
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([float(m[1]) for m in matches]) / num_t
    print('Average Mallet LDA topic coherence: %.4f.' % avg_topic_coherence)
    return avg_topic_coherence

# CFMC: Nueva version para considerar bigramas y Mallet LDA

def lda(abs, Clusters, outputpath, start_tpcs, end_tpcs):
    mallet_path = "/home/cmendezc/Documents/ccg/gitlab-biomedical-literature-explorer/CIFN-CCG40/source/mallet-2.0.8/bin/mallet"
    # ShinyApp: mallet_path = "../source/mallet-2.0.8/bin/mallet"
    start_tpcs = int(start_tpcs)
    end_tpcs = int(end_tpcs)
    # CFMC: Input abs_lemmatized.txt
    t0 = time()
    stop_words = stopwords.words('english')
    # print(stop_words)
    # First we will create a list with all the abstracts and a diccionary with all the words.
    temp_documents = []
    wordcount = {}
    with open(abs, encoding='utf8', mode='r') as abstracts_tot:
        for line in abstracts_tot:
            ln = line.split("\t")
            # CFMC: We filter punctuation and numbers here instead of after training
            # Original: documents.append(ln[2])
            # CFMC: To append title to abstract
            tit_abs = ln[1] + " " + ln[2]
            ln_clean = strip_punctuation(tit_abs)
            ln_clean = strip_numeric(ln_clean)
            ln_clean = strip_short(ln_clean)
            ln_clean = [word for word in simple_preprocess(ln_clean) if word not in stop_words]
            temp_documents.append(ln_clean)
            # CFMC - End
            # Original: for word in ln[2].split():
            for word in ln_clean:
                if word not in stop_words:
                    if word not in wordcount:
                        wordcount[word] = 1
                    else:
                        wordcount[word] += 1
    print("Total documents: {}".format(len(temp_documents)))
    # Removing the words that are at least in half of the abstracts
    # CFMC: I will test without this filter
    # Original: for word in wordcount:
    # Original: if wordcount[word] > round(len(documents)/2):
    # Original: stop_words.append(word)

    # Removing the words that are at least in half of the abstracts
    documents = []
    # print("Start Len temp_documents: {}".format(len(temp_documents)))
    for i in range(len(temp_documents)):
        # print("Start Len temp_documents[i]: {}".format(len(temp_documents[i])))
        temp_line = []
        for w in range(len(temp_documents[i])):
            if wordcount[temp_documents[i][w]] > round(len(temp_documents)/2):
                pass
                # print("Borrar: {}".format(temp_documents[i][w]))
            else:
                temp_line.append(temp_documents[i][w])
        documents.append(temp_line)
        # print("End Len documents[i]: {}".format(len(documents[i])))
    # print("End Len documents: {}".format(len(documents)))
    #quit()

    # Since we have multiple clusters, we will make a list with the corresponding cluster each abstract has.
    clusters = []
    with open(Clusters, encoding='utf8', mode='r') as clusters_tot:
        cluster_tot = iter(clusters_tot)
        next(clusters_tot)
        for line in clusters_tot:
            ln = line.split()
            clusters.append(int(ln[2]))
        k = max(
            clusters) + 1  # this variable will determine the number the clusters that are in our data, so it will tell us how many models we need to do.
    #print("clusters: {}".format(clusters))
    #quit()
    # Now we need to create a dictionary were each cluster has their corresponding abstracts.
    abstracts = {}
    for i in range(k):
        abstracts_cl = []
        for j in range(len(documents)):
            if clusters[j] == i:
                abstracts_cl.append(documents[j])
        abstracts[i] = abstracts_cl
        bigram = gensim.models.Phrases(abstracts[i], min_count=1, threshold=1)
        ###bigram_mod = gensim.models.phrases.Phraser(bigram)
        ###print(bigram_mod)
        # CFMC: Add bigrams to docs
        for idx in range(len(abstracts[i])):
            bigrams = bigram[abstracts[i][idx]]
            # print(bigrams)
            for token in bigrams:
                # print("Token bigram: {}".format(token))
                if '_' in token:
                    # Token is a bigram, add to document.
                    abstracts[i][idx].append(token)
    # print(abstracts[0]) # Lista de abstracts del cluster

    # During the performance of the model I will create a dictionary with the supplementary_data from each cluster.
    results = {}
    mallet_top_docs = {}
    num_words = 10
    for num_topics in range(start_tpcs, end_tpcs + 1):
        for lda_package in ["gensim", "mallet"]:
            # We will notify the user that the model is already starting.
            print("\nPERFORMING LDA MODEL: {} TOPICS WITH {}".format(num_topics, lda_package))
            for i in range(k):
                print("Processing cluster {} of {}".format(i + 1, k))
                result = []
                # CFMC: This tutorial uses min_count=20 and no threshold (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)
                # Original: bigram = gensim.models.Phrases(abstracts[i], min_count = 5, threshold = 100)
                if lda_package == "gensim":
                    print("Gensim LDA topic modeling...")
                    # print(abstracts[i])
                    # Original: bigram_mod = gensim.models.phrases.Phraser(bigram)
                    # CFMC: Creo que ya no es necesario quitar stopwords porque ya se hizo arriba y tampoco volver a tomar los bigramas
                    # Original: data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in abstracts[i]]
                    # Original: data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
                    # Original: id2word = gensim.corpora.Dictionary(data_words_bigrams)
                    # CFMC: Crear dictionary
                    id2word = gensim.corpora.Dictionary(abstracts[i])
                    # CFMC: This tutorial filter out words that occur less than 20 documents, or more than 50% of the documents (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html).
                    # CFMC: id2word.filter_extremes(no_below=20, no_above=0.5)
                    # CFMC: We test without no_above threshold as we consider high frequent words
                    ### CFMC: We test without filter: id2word.filter_extremes()
                    # Original: texts = data_words_bigrams
                    # Original: corpus = [id2word.doc2bow(text) for text in texts]
                    # CFMC: Bag of words representation
                    corpus = [id2word.doc2bow(text) for text in abstracts[i]]
                    # print('Number of unique tokens: %d' % len(id2word))
                    # print('Number of documents: %d' % len(corpus))
                    # Training
                    # CFMC: This tutorial doesn't create Tfidf model (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html).
                    # Original: tfidf = models.TfidfModel(corpus)
                    # Original: lda_model = gensim.models.LdaMulticore(tfidf[corpus], num_topics=1, id2word=id2word, passes=10, workers=4)
                    # CFMC: We followed parameters of this tutorial (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html).
                    # Set training parameters.
                    chunksize = 2000
                    passes = 20
                    iterations = 400
                    eval_every = None  # Don't evaluate model perplexity, takes too much time.

                    # CFMC: lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=id2word, passes=passes,
                    # CFMC:                                    workers=6, iterations=iterations)
                    tfidf = models.TfidfModel(corpus)
                    lda_model = gensim.models.LdaMulticore(tfidf[corpus], num_topics=num_topics, id2word=id2word, passes=passes,
                                                           workers=6, iterations=iterations)
                    lda_topics = lda_model.show_topics(num_words=15)
                    # CFMC: Average topic coherence
                    # top_topics = lda_model.top_topics(corpus)
                    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
                    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
                    # print('Average topic coherence: %.4f.' % avg_topic_coherence)
                    # CFMC End

                    # from pprint import pprint
                    # pprint(top_topics)

                    # CFMC: Following filter may be done over input abstracts, not here after training
                    # Original: filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

                    # CFMC: Format topics to output

                    formatted_topics = get_topics_gensim(lda_topics, num_topics, num_words)
                    # print("formatted_topics: {}".format(formatted_topics))
                    result.append(formatted_topics)

                    # CFMC: perplexity = lda_model.log_perplexity(corpus)
                    perplexity = lda_model.log_perplexity(tfidf[corpus])
                    # print('Perplexity: %.4f.' % perplexity)
                    result.append(perplexity)
                    # Original: coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                    #                                     coherence='c_v')
                    coherence_model_lda = CoherenceModel(model=lda_model, texts=abstracts[i], dictionary=id2word,
                                                         coherence='c_v')
                    coherence = coherence_model_lda.get_coherence()
                    result.append(coherence)
                    # print('Coherence: %.4f.' % coherence)

                elif lda_package == "mallet":
                    print("Mallet LDA topic modeling...")
                    # CFMC: Documentation in https://github.com/maria-antoniak/little-mallet-wrapper
                    # CFMC: We followed this tutorial: https://github.com/maria-antoniak/little-mallet-wrapper/blob/master/demo.ipynb
                    path_to_mallet = mallet_path
                    doc = ""
                    docs = []
                    for t in abstracts[i]:
                        # print("t: {}".format(t))
                        doc = ' '.join(t)
                        docs.append(doc)
                    # print("docs: {}".format(docs))
                    # process_string: A simple string processor that prepares raw text for topic modeling.
                    # training_data = [lmw.process_string(t) for t in docs]
                    training_data = docs
                    # print(training_data)
                    training_data = [d for d in training_data if d.strip()]
                    print("len(training_data): {}".format(len(training_data)))
                    # print_dataset_stats: Displays basic statistics about the training dataset.
                    lmw.print_dataset_stats(training_data)
                    # quick_train_topic_model: Imports training data, trains an LDA topic model using MALLET, and returns the topic keys and document distributions.
                    # RETURNS: The 20 most probable words for each topic.
                    # RETURNS: Topic distribution (list of probabilities) for each document.
                    # topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet,
                    #                                                              outputpath,
                    #                                                              num_topics,
                    #                                                              training_data)

                    path_to_training_data = outputpath + '/training.txt'
                    path_to_formatted_training_data = outputpath + '/mallet.training'
                    path_to_model = outputpath + '/mallet.model.' + str(num_topics)
                    path_to_topic_keys = outputpath + '/mallet.topic_keys.' + str(num_topics)
                    path_to_topic_distributions = outputpath + '/mallet.topic_distributions.' + str(num_topics)
                    path_to_word_weights = outputpath + '/mallet.word_weights.' + str(num_topics)
                    path_to_diagnostics = outputpath + '/mallet.diagnostics.' + str(num_topics) + '.xml'

                    lmw.import_data(path_to_mallet,
                                    path_to_training_data,
                                    path_to_formatted_training_data,
                                    training_data)
                    lmw.train_topic_model(path_to_mallet,
                                          path_to_formatted_training_data,
                                          path_to_model,
                                          path_to_topic_keys,
                                          path_to_topic_distributions,
                                          path_to_word_weights,
                                          path_to_diagnostics,
                                          num_topics)

                    #assert(len(topic_distributions) == len(training_data))
                    topic_keys = lmw.load_topic_keys(outputpath + '/mallet.topic_keys.' + str(num_topics))
                    print("Topics: ")
                    for x, t in enumerate(topic_keys):
                        print(x, '\t', ' '.join(t[:num_words]))
                    # To format topics
                    formatted_topics = get_topics_mallet(topic_keys, num_topics, num_words)
                    # print("formatted_topics: {}".format(formatted_topics))
                    result.append(formatted_topics)

                    # get_top_docs: Gets the documents with the highest probability for the target topic.
                    # RETURNS 	list of tuples (float, string)
                    #   The topic probability and document text for the n documents
                    #   with the highest probability for the target topic.
                    topic_distributions = lmw.load_topic_distributions(
                        outputpath + '/mallet.topic_distributions.' + str(num_topics))

                    # print("Top documents: ")
                    for top in range(num_topics):
                        for p, d in lmw.get_top_docs(training_data, topic_distributions, topic_index=top, n=3):
                            # print(round(p, 4), d)
                            # print()
                            if i in mallet_top_docs:
                                if top in mallet_top_docs[i]:
                                    mallet_top_docs[i][top].append("{}\t{}".format(round(p, 4), d))
                                else:
                                    mallet_top_docs[i] = {top: ["{}\t{}".format(round(p, 4), d)]}
                            else:
                                mallet_top_docs = {i: {top: ["{}\t{}".format(round(p, 4), d)]}}

                    # load_topic_word_distributions: Loads the topic word distributions. These are the probabilities for each word for each topic.
                    # RETURNS 	defaultdict of defaultdict of float
                    #   Map of topics to words to probabilities.
                    topic_word_probability_dict = lmw.load_topic_word_distributions(outputpath + '/mallet.word_weights.' + str(num_topics))
                    for _topic, _word_probability_dict in topic_word_probability_dict.items():
                        print('Topic', _topic)
                        for _word, _probability in sorted(_word_probability_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(round(_probability, 4), '\t', _word)
                        print()

                    perplexity = 0.0
                    result.append(perplexity)
                    # CFMC: coherence. This metric measures whether the words in a topic tend to co-occur together.
                    #   We add up a score for each distinct pair of top ranked words.
                    #   The score is the log of the probability that a document containing
                    #   at least one instance of the higher-ranked word also contains
                    #   at least one instance of the lower-ranked word.
                    #   Since these scores are log probabilities they are negative.
                    #   Large negative values indicate words that don't co-occur often;
                    #   values closer to zero indicate that words tend to co-occur more often.
                    #   https://mallet.cs.umass.edu/diagnostics.php
                    coherence = get_mallet_coherence(path_to_diagnostics, num_topics)
                    result.append(coherence)

                results[i] = result
            # print(supplementary_data)
                    # quit()
                    # get_js_divergence_topics: Calculates the Jensen-Shannon divergence between the two target topic distributions.
                    # RETURNS 	float 	Jensen-Shannon divergence of the requested topic distributions.
                    # lmw.get_js_divergence_topics(0, 7, topic_word_probability_dict)

            name = Clusters
            name = name.split("/")
            filename1 = lda_package + "LDA_" + str(num_topics) + "tpcs_" + name[len(name) - 1]
            filename2 = lda_package + "LDA_clst_" + str(num_topics) + "tpcs_" + name[len(name) - 1]
            if lda_package == "mallet":
                filename3 = lda_package + "LDA_top_docs_" + str(num_topics) + "tpcs_" + name[len(name) - 1]
                dump(mallet_top_docs, os.path.join(outputpath, filename3.replace(".txt", ".joblib")))

            # To obtain words per topic to display, we prefer 15
            # num_words = 15 / num_topics
            with open(os.path.join(outputpath, filename1), mode='w', encoding='utf8') as oFile:
                for j in range(len(documents)):
                    for i in range(k):
                        if clusters[j] == i:
                            topics = results[i][0]
                            # CFMC: Format topics is previously done
                            '''
                            print(topics)
                            topics = str(topics)[1:-1]
                            print(topics)
                            topics = str(topics)[1:-1]
                            print(topics)
                            topics = topics.replace('0', "")
                            print(topics)
                            topics = topics.replace("+", "\t")
                            print(topics)
                            # topics = topics.replace('.1*', "")
                            # topics = topics.replace('.2*', "")
                            # topics = topics.replace('.3*', "")
                            # topics = topics.replace('.4*', "")
                            # topics = topics.replace('.5*', "")
                            # topics = topics.replace('.6*', "")
                            # topics = topics.replace('.7*', "")
                            # topics = topics.replace('.8*', "")
                            # topics = topics.replace('.9*', "")
                            topics = sub(r"\.\d+\*", "", topics)
                            print(topics)
                            topics = topics.replace('"', "")
                            topics = topics.replace(',', "")
                            topics = topics.replace("'", "")
                            # topics = sub(r"\)\s\([0-9]+\s", "\t", topics)
                            '''
                            oFile.write("{}\n".format(topics))

            with open(os.path.join(outputpath, filename2), mode='w', encoding='utf8') as cpFile:
                for i in range(k):
                    topics = results[i][0]
                    # CFMC: Format topics is previously done
                    '''
                    topics = str(topics)[1:-1]
                    topics = str(topics)[1:-1]
                    topics = topics.replace('0', "")
                    topics = topics.replace("+", "\t")
                    # topics = topics.replace('.1*', "")
                    # topics = topics.replace('.2*', "")
                    # topics = topics.replace('.3*', "")
                    # topics = topics.replace('.4*', "")
                    # topics = topics.replace('.5*', "")
                    # topics = topics.replace('.6*', "")
                    # topics = topics.replace('.7*', "")
                    # topics = topics.replace('.8*', "")
                    # topics = topics.replace('.9*', "")
                    topics = sub(r"\.\d+\*", "", topics)
                    topics = topics.replace('"', "")
                    topics = topics.replace(',', "")
                    topics = topics.replace("'", "")
                    # topics = topics.replace(" \t ", " ")
                    # topics = sub(r"\)\s\([0-9]+\s", " ", topics)
                    '''
                    # We add perplexity and coherence
                    perplexity = results[i][1]
                    coherence = results[i][2]
                    cpFile.write("{}\t{}\t{}\t{}\n".format(i, topics, perplexity, coherence))

    print("\nTime used for all the clusters: %fs \n" % (time() - t0))

# Local
# path = "/home/cmendezc/Documents/ccg/gitlab-biomedical-literature-explorer/CIFN-CCG40/ShinyApp/sessions/ccg_manager/pmid-acinetobac-set-vTSNESVDNewLSA"
path = "/home/cmendezc/Documents/ccg/gitlab-proyecto-ab"
input = os.path.join(path, "source/abs_lemmatized.txt")
clus = os.path.join(path, "supplementary_data/clusters/k113_perp30.txt")
output = os.path.join(path, "supplementary_data/LDA")
# mallet_path = "/home/cmendezc/Documents/ccg/gitlab-biomedical-literature-explorer/CIFN-CCG40/source/mallet-2.0.8/bin/mallet"
lda(input, clus, output, 1, 10)