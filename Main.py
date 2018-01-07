from Utils import *
from EM import EM
import time

def em_initialization(articles, num_of_articles):
    clustered_articles = list()
    for i, article in enumerate(articles):
        clustered_articles.append([0] * num_of_articles)
        clustered_articles[i][i % 9] = 1
    return clustered_articles


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
    articles = read_file(train_file, parse_sep_articles, " ")
    articles, vocab_size = filter_rare_words(train_file, articles)
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters, vocab_size)
    while True:
        time0 = time.time()
        likelihood = em.calculate_likelihood()
        print likelihood
        em.update_parameters()
        time1 = time.time()
        print time1 - time0
