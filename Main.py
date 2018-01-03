from Utils import *
from EM import EM


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
    print "Vocabulary size:", len(filtered_words)
    return articles


if __name__ == "__main__":
    train_file = "data/develop.txt"
    num_of_topics = 9
    articles = read_file(train_file, parse_sep_articles, " ")
    articles = filter_rare_words(train_file, articles)
    clusters = em_initialization(articles, num_of_topics)
    em = EM(num_of_topics, articles, clusters)
    em.update_parameters()
