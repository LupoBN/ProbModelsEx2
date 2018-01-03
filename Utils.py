# Parse the data file.
from collections import Counter


def parse_no_title(lines, seperator):
    words = list()
    for i in range(2, len(lines), 4):
        parsed_line = lines[i].split(seperator)
        parsed_line.remove("")
        for word in parsed_line:
            words.append(word)
    return words

def parse_sep_articles(lines, seperator):
    articles = list()
    for i in range(2, len(lines), 4):
        article = Counter()
        parsed_line = lines[i].split(seperator)
        parsed_line.remove("")
        article.update(parsed_line)
        articles.append(article)
    return articles

# Read the data file.
def read_file(file_name, parse_func, separator=None):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    file.close()
    return parse_func(lines, separator)


# Write the output file.
def write_file(file_name, content):
    file = open(file_name, 'w')
    file.write(content)
    file.close()

