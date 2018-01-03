from Utils import *


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
    return articles


if __name__ == "__main__":
    train_file = "data/develop.txt"
    articles = read_file(train_file, parse_sep_articles, " ")
    articles = filter_rare_words(train_file, articles)
    clusters = em_initialization(articles)
