# Matan Ben Noach Itay Mosafi 201120441 205790983


# Calculate the perplexity of a given estimator on the validation data.
def calculate_perplexity(likelihood, validation):
    log_likelihood = likelihood
    # Take the negative average of the log likelihood.
    log_likelihood /= -len(validation)
    # Return 2 in power of the log likelihood to retrieve the perplexity.
    return pow(2, log_likelihood)

import matplotlib.pyplot as plt
import numpy as np
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