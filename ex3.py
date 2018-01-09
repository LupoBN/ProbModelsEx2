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
    filtered_articles = [Counter(word for word in article.elements() if frequencies[word] > 3) for article in articles]
    filtered_words = [word for word in words if frequencies[word] > 3]
    filtered_words_set = set()
    for article in filtered_articles:
        filtered_words_set.update(set(article))
    vocab_size = len(filtered_words_set)
    print "Vocabulary size:", vocab_size
    return filtered_articles, vocab_size, filtered_words


def EM_Algorithm(em, list_of_words):
    em.update_parameters()

    likelihoods = [int(em.calculate_likelihood())]
    perplexities = [calculate_perplexity(em, list_of_words)]
    # EM algorithm.
    while True:

        em.update_parameters()
        likelihood = int(em.calculate_likelihood())
        perplexity = calculate_perplexity(em, list_of_words)
        print "Likelihood:", likelihood
        print "Perplexity", perplexity
        print "Accuracy: " + str(em.calculate_accuracy(topics, article_topics))
        likelihoods.append(likelihood)
        perplexities.append(perplexity)
        if likelihoods[-1] == likelihoods[-2]:
            break

        assert likelihoods[-1] >= likelihoods[-2]

    return likelihoods, perplexities


def create_confusion_matrix(articles, article_topics):
    conf_mat = [[0] * 10 for i in range(0, 9)]
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
    articles, vocab_size, list_of_words = filter_rare_words(articles, list_of_words)
    # Cluster the articles according to the initialization instructions.
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters, vocab_size)
    likelihoods, perplexities = EM_Algorithm(em, list_of_words)
    conf_mat = create_confusion_matrix(articles, article_topics)
    list_of_topics = sorted(topics, key=topics.get)
    plot_results(likelihoods, "Likelihood Graph", "Likelihood")
    plot_results(perplexities, "Perplexity Graph", "Perplexity")
    for histogram in conf_mat:
        create_histogram(histogram[:-1], list_of_topics)
