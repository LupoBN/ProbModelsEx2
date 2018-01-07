# Matan Ben Noach Itay Mosafi 201120441 205790983
import matplotlib.pyplot as plt
import numpy as np
import math
# Calculate the perplexity of a given estimator on the validation data.
def calculate_perplexity(estimator, validation):
    log_likelihood = 0.0
    for word in validation:
        # Estimate the probability of a word in the validation set.
        prob = estimator.get_word_prob(word)
        try:
            # Calculate the log likelihood for a word in the validation set.
            log_likelihood += math.log(prob, 2)
        except:
            log_likelihood += -float("Inf")
    # Take the negative average of the log likelihood.
    log_likelihood /= -len(validation)
    # Return 2 in power of the log likelihood to retrieve the perplexity.
    return pow(2, log_likelihood)

def create_histogram(frequencies, topics):
    x_pos = [i for i in range(len(topics))]
    plt.bar(x_pos, frequencies, align='center')
    plt.xticks(x_pos, topics)
    plt.ylabel('Number of articles')
    maximum = np.argmax(frequencies)
    plt.title(topics[maximum] + " Cluster")
    plt.show()

# Plots the result of the training.
def plot_results(history, title, ylabel, xlabel='Iteration number'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(history)
    plt.show()