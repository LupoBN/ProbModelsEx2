class EM(object):
    def __init__(self, num_of_topics, articles):
        self._alphas = list()
        self._P = list()
        for i in range(0, num_of_topics):
            self._alphas.append(1.0 / float(num_of_topics))
        


