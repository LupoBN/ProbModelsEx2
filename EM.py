from collections import defaultdict

import math

EPSILON = 0.0001
LAMBDA = 0.05
K = 10

class EM(object):
    def __init__(self, num_of_topics, articles, article_clusters, vocab_size):
        self._alphas = list()
        self._P = list()
        self._one_divided_by_N = 1.0 / float(len(articles))
        self._ntk = articles
        self._nt = list()
        self._article_clusters = article_clusters
        self._vocab_size = vocab_size # needed for smoothing
        self._initialize_nt(articles)
        self._initialize_parameters(num_of_topics, articles, article_clusters)

    def _initialize_nt(self, articles):
        for article in articles:
            self._nt.append(len(article))

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

    def _calculate_z(self, t):
        z = [0.0] * len(self._alphas)
        for i in range(0, len(self._alphas)):
            z[i] = math.log(self._alphas[i])
            for word in self._P[i]:
                z[i] += self._ntk[t][word] * math.log(self._P[i][word])
        return z

    def _calculate_stable_wti(self, z, i, m):
        numerator = math.exp(z[i] - m)
        denominator = 0.0
        for j in range(0, len(z)):
            if z[j] - m >= -K:
                denominator += math.exp(z[j] - m)
        return numerator / denominator

    def _calculate_wti_numerator(self, i, t):
        # TODO: Add the smoothing and underflow control.
        z = self._calculate_z(t)
        m = max(z)
        if z[i] - m < -K:
            wti = 0.0
        else:
            wti = self._calculate_stable_wti(z, i, m)
        return wti

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

    def _update_P(self, w):
        for i in range(0, len(self._P)):
            for word in self._P[i]:
                numerator = 0.0
                denominator = 0.0
                for t in range(0, len(w)):
                    numerator += w[t][i] * self._ntk[t][word]
                    denominator += w[t][i] * self._nt[t]
                numerator += LAMBDA
                denominator += self._vocab_size * LAMBDA
                self._P[i][word] = numerator / denominator

    def update_parameters(self):
        w = list()
        for t in range(0, len(self._ntk)):
            w.append(list())
            for i in range(0, len(self._alphas)):
                wti = self._calculate_wti_numerator(i, t)
                w[t].append(wti)
        for t in range(0, len(w)):
            # alpha_j_sum = sum(w[i]) #should be t?
            alpha_j_sum = sum(w[t])
            if alpha_j_sum <= 0.000001:
                continue
            for i in range(0, len(self._alphas)):  # shouldn't it be done on the new alphas?
                w[t][i] /= alpha_j_sum
        self._update_alphas(w)
        self._update_P(w)
