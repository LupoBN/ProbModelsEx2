from collections import defaultdict

import math


class EM(object):
    def __init__(self, num_of_topics, articles, article_clusters):
        self._alphas = list()
        self._P = list()
        self._one_divided_by_N = 1.0 / float(len(articles))
        self._ntk = articles
        self._article_clusters = article_clusters
        self._initialize_parameters(num_of_topics, articles, article_clusters)

    def _initialize_parameters(self, num_of_topics, articles, article_clusters):
        for i in range(0, num_of_topics):
            self._alphas.append(1.0 / float(num_of_topics))
            self._P.append(defaultdict(lambda: 0.0))
        self._count_cluster_words(articles, article_clusters)
        self._initialize_word_probs()

    def _count_cluster_words(self, articles, article_clusters):
        for article, one_hot_vec in zip(articles, article_clusters):
            indices = [i for i, x in enumerate(one_hot_vec) if x == 1]
            for word in article:
                for index in indices:
                    self._P[index][word] += 1.0

    def _initialize_word_probs(self):
        for cluster in self._P:
            total_cluster_words = float(sum(cluster.values()))
            for word in cluster:
                cluster[word] /= total_cluster_words

    def _calculate_wti_numerator(self, i, t):
        # TODO: Add the smoothing.
        wti = self._alphas[i]
        for word in self._P[i]:
            wti *= math.pow(self._P[i][word], self._ntk[t][word])
        return wti

    def _update_alphas(self, w):
        for i in range(0, len(self._alphas)):
            self._alphas[i] = self._one_divided_by_N
            for t in range(0, len(w)):
                self._alphas[i] *= w[t][i]

    def _update_P(self, w):
        for i in range(0,len(self._P)):
            for word in self._P[i]:
                numerator = 0.0
                denominator = 0.0
                for t in range(0, len(w)):
                    numerator += w[t][i] * self._ntk[t][word]
                    denominator += w[t][i] * sum(self._ntk[t].values())

                self._P[i][word] = numerator / denominator

    def update_parameters(self):
        w = list()
        for t in range(0, len(self._ntk)):
            w.append(list())
            for i in range(0, len(self._alphas)):
                wti = self._calculate_wti_numerator(i, t)
                w[t].append(wti)
        for t in range(0, len(w)):
            alpha_j_sum = sum(w[i])
            for i in range(0, len(self._alphas)):
                w[t][i] /= alpha_j_sum
        self._update_alphas(w)
        self._update_P(w)
