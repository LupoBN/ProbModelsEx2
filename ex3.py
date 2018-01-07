# Matan Ben Noach Itay Mosafi 201120441 205790983
from Utils import *
from Helpers import *
from EM import EM

# Initialize the clusters according to the instructions.
def em_initialization(articles, num_of_articles):
    clustered_articles = list()
    for i, article in enumerate(articles):
        clustered_articles.append([0] * num_of_articles)
        clustered_articles[i][i % 9] = 1
    return clustered_articles


# Filter rare words.
def filter_rare_words(articles, words):
    frequencies = Counter()
    frequencies.update(words)
    articles = [Counter(word for word in article.elements() if frequencies[word] > 3) for article in articles]
    filtered_words = set()
    for article in articles:
        filtered_words.update(set(article))
    vocab_size = len(filtered_words)
    print "Vocabulary size:", vocab_size
    return articles, vocab_size

def EM_Algorithm(em, list_of_words):
    likelihood = -float("Inf")
    last_liklihood = 1.0
    likelihoods = list()
    perplexities = list()
    counter = 0
    # EM algorithm.
    while abs(last_liklihood - likelihood) > 0.01:
        last_liklihood = likelihood
        likelihood = em.calculate_likelihood()
        perplexity = calculate_perplexity(em, list_of_words)
        print "Likelihood:", likelihood
        likelihoods.append(likelihood)
        perplexities.append(perplexity)
        em.update_parameters()
        counter += 1
        if counter == 10:
            break
    return likelihoods, perplexities

def create_confusion_matrix(articles, article_topics):
    conf_mat = [[0] * 10 for i in range(0,9)]
    clustered_articles = em.cluster_articles(articles)

    for i in range(len(clustered_articles)):
        article_cluster = clustered_articles[i].index(1)
        topics_ind = [j for j, x in enumerate(article_topics[i]) if x == 1]
        for ind in topics_ind:
            conf_mat[article_cluster][ind] += 1
        conf_mat[article_cluster][-1] += 1
    return conf_mat





if __name__ == "__main__":

    train_file = "data/develop.txt"
    topics_file = "data/topics.txt"
    num_of_topics = 9
    # Read the articles and get the histograms of words for each article.
    articles = read_file(train_file, parse_sep_articles, " ")
    list_of_words = read_file(train_file, parse_no_title, " ")
    topics = read_file(topics_file, parse_topics)
    article_topics = read_file(train_file, parse_titile, "\t", topics)
    # Filter rare words.
    articles, vocab_size = filter_rare_words(articles, list_of_words)
    # Cluster the articles according to the initialization instructions.
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters, vocab_size)
    likelihoods, perplexities = EM_Algorithm(em, list_of_words)
    print "Accuracy: " + str(em.calculate_accuracy(article_topics))
    conf_mat = create_confusion_matrix(articles, article_topics)
    list_of_topics = sorted(topics, key=topics.get)
    plot_results(likelihoods, "Likelihood Graph", "Likelihood")
    plot_results(perplexities, "Perplexity Graph", "Perplexity")
    for histogram in conf_mat:
        create_histogram(histogram[:-1], list_of_topics)
