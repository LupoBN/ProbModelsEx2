# Matan Ben Noach Itay Mosafi 201120441 205790983
from Utils import *
from EM import EM

# Initialize the clusters according to the instructions.
def em_initialization(articles, num_of_articles):
    clustered_articles = list()
    for i, article in enumerate(articles):
        clustered_articles.append([0] * num_of_articles)
        clustered_articles[i][i % 9] = 1
    return clustered_articles


# Filter rare words.
def filter_rare_words(develop_file, articles):
    words = read_file(develop_file, parse_no_title, " ")
    frequencies = Counter()
    frequencies.update(words)
    articles = [Counter(word for word in article.elements() if frequencies[word] > 3) for article in articles]
    filtered_words = set()
    for article in articles:
        filtered_words.update(set(article))
    vocab_size = len(filtered_words)
    print "Vocabulary size:", vocab_size
    return articles, vocab_size


if __name__ == "__main__":

    train_file = "data/develop.txt"
    num_of_topics = 9
    # Read the articles and get the histograms of words for each article.
    articles = read_file(train_file, parse_sep_articles, " ")
    # Filter rare words.
    articles, vocab_size = filter_rare_words(train_file, articles)
    # Cluster the articles according to the initialization instructions.
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters, vocab_size)
    last_liklihood = 0.0
    likelihood = 2.0
    # EM algorithm.
    while abs(last_liklihood - likelihood) > 0.000001:
        last_liklihood = likelihood
        likelihood = em.calculate_likelihood()
        print likelihood
        em.update_parameters()
