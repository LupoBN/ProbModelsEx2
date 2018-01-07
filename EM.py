# Matan Ben Noach Itay Mosafi 201120441 205790983

from collections import defaultdict
import numpy as np

import math

EPSILON = 0.001 #0.005 #0.0001
LAMBDA = 0.1#0.1 #0.05
K = 10


class EM(object):
    def __init__(self, num_of_topics, articles, article_clusters, vocab_size):
        self._alphas = list()
        self._P = list()
        self._one_divided_by_N = 1.0 / float(len(articles))
        self._ntk = articles
        self._nt = list()
        self._article_clusters = article_clusters
        self._vocab_size = vocab_size  # needed for smoothing
        self._initialize_nt(articles)
        self._initialize_parameters(num_of_topics, articles, article_clusters)
        self._last_wti = None

    # Initialize the nt according to each length of an article.
    def _initialize_nt(self, articles):
        for article in articles:
            self._nt.append(len(article))


    # Initialize the parameters according to
    def _initialize_parameters(self, num_of_topics, articles, article_clusters):
        for i in range(0, num_of_topics):
            self._alphas.append(1.0 / float(num_of_topics))
            self._P.append(defaultdict(lambda: LAMBDA / (LAMBDA * self._vocab_size)))
        self._count_cluster_words(articles, article_clusters)
        self._initialize_word_probs()

    def get_word_prob(self, word):
        prob = 0.0
        for i in range(len(self._alphas)):
            prob += self._alphas[i] * self._P[i][word]
        return prob

    # Initialize the P values.
    def _count_cluster_words(self, articles, article_clusters):
        for article, one_hot_vec in zip(articles, article_clusters):
            indices = [i for i, x in enumerate(one_hot_vec) if x == 1]
            for word in article:
                for index in indices:
                    self._P[index][word] += 1.0

    # Initialize the word probs.
    def _initialize_word_probs(self):
        for cluster in self._P:
            total_cluster_words = float(sum(cluster.values()))
            for word in cluster:
                cluster[word] /= total_cluster_words

    # Calculate the z value for the underflow management.
    def _calculate_z(self, t):
        z = [0.0] * len(self._alphas)
        for i in range(0, len(self._alphas)):
            z[i] = math.log(self._alphas[i])
            for word in self._ntk[t]:
                z[i] += self._ntk[t][word] * math.log(self._P[i][word])
        return z

    # Calculate the wti with the underflow management.
    def _calculate_stable_wti(self, z, i, m):
        numerator = math.exp(z[i] - m)
        """
        This code is not needed.
         
        denominator = 0.0
        for j in range(0, len(z)):
            #if z[j] - m >= -K:
                #denominator += math.exp(z[j] - m)
        return numerator / denominator
        
        """
        return numerator

    # Calculate the numerator of wti.
    def _calculate_wti_numerator(self, z, m, i):
        #z = self._calculate_z(t)
        #m = max(z)
        if z[i] - m < -K:
            wti = 0.0
        else:
            wti = self._calculate_stable_wti(z, i, m)
        return wti

    # Update the alpha values.
    def _update_alphas(self, w):
        for i in range(0, len(self._alphas)):
            # self._alphas[i] = self._one_divided_by_N
            temp_sum = 0.0
            for t in range(0, len(w)):
                temp_sum += w[t][i]
                # self._alphas[i] *= w[t][i] # should be sum?
            new_alpha_i = self._one_divided_by_N * temp_sum
            self._alphas[i] = new_alpha_i if new_alpha_i > 0 else EPSILON
        alpha_sum = sum(self._alphas)
        for i in range(0, len(self._alphas)):
            self._alphas[i] /= alpha_sum

    # Update the P values.
    def _update_P(self, w):
        for i in range(0, len(self._P)):
            numerators = defaultdict(lambda: LAMBDA)
            denominators = defaultdict(lambda: self._vocab_size * LAMBDA)
            for t in range(0, len(w)):
                for word in self._ntk[t]:
                    numerators[word] += w[t][i] * self._ntk[t][word]
                    denominators[word] += w[t][i] * self._nt[t]
            for word in self._P[i]:
                self._P[i][word] = numerators[word] / denominators[word]

    # Calculate the likelihood.
    def calculate_likelihood(self):
        total_ln_l = 0.0
        for t in range(0, len(self._ntk)):
            #z = []
            z = self._calculate_z(t)
            """
            for i in range(0, len(self._alphas)):
                k_sum = 0.0
                for word in self._ntk[t]:
                    k_sum += self._ntk[t][word] * math.log(self._P[i][word])
                z.append(math.log(self._alphas[i]) + k_sum)
            """
            m = max(z)

            e_sum = 0.0
            for j in range(0, len(z)):
                if z[j] - m >= -K:
                    e_sum += math.exp(z[j] - m)

            total_ln_l += m + math.log(e_sum)

        return total_ln_l

    def calculate_accuracy(self, articles_topics):
        correct_predictions = 0.0
        for t in xrange(len(articles_topics)):
            max_value_index = np.asarray(self._last_wti[t]).argsort()[-1:]
            real_indexes = [i for i, x in enumerate(articles_topics[t]) if x == 1]
            if max_value_index in real_indexes:
                correct_predictions += 1
        return correct_predictions / len(self._article_clusters)

    # Update parameters.
    def update_parameters(self):
        w = list()
        for t in range(0, len(self._ntk)):
            z = self._calculate_z(t)
            m = max(z)
            w.append(list())
            for i in range(0, len(self._alphas)):
                wti = self._calculate_wti_numerator(z, m, i)
                w[t].append(wti)
        for t in range(0, len(w)):
            alpha_j_sum = sum(w[t])
            if alpha_j_sum <= 0.000001:
                continue
            for i in range(0, len(self._alphas)):
                w[t][i] /= alpha_j_sum
        self._last_wti = w
        self._update_alphas(w)
        self._update_P(w)

    def cluster_articles(self, articles):
        article_clusters = list()
        for t, article in enumerate(articles):
            article_clusters.append([0] * 9)
            best = -float("Inf")
            best_cluster = -1

            for i in range(len(self._alphas)):
                cluster_prob = self._alphas[i]
                for word in article:
                    cluster_prob *= article[word] * self._P[i][word]
                if cluster_prob > best:
                    best = cluster_prob
                    best_cluster = i
            article_clusters[t][best_cluster] = 1
        return article_clusters








