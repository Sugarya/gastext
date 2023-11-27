from typing import Any
import numpy as np
from scipy.special import softmax
from metrics import query_increase


class LikelihoodEstimator:

    def __init__(self, victim_model):
        self._victim_model = victim_model

    def _init_estimate(self, probs, label):
        self._attack_probs = probs
        self._attack_label = label

    def _set_real_label(self,label):
        self._real_label = label
        

    def get_likelihood_score(self, sentence_list):
        probs, label = self.get_predict(sentence_list)
        score = probs[label] - self._attack_probs[self._attack_label]
        return label == self._attack_label, score

    def get_predict(self, sentence_list):
        logits = self._victim_model(sentence_list)
        # print("LikelihoodEstimator get_predict logits = {}".format(logits))
        # probs = softmax(logits)
        # print("LikelihoodEstimator get_predict probs = {}".format(probs))
        predict_label = np.argmax(logits)
        query_increase()
        return logits, predict_label

