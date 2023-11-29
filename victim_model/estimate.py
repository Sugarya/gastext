from typing import Any
import numpy as np
from scipy.special import softmax
from metrics import query_increase


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def valid_label(victim_model, text, real_label):
    prob_label = _get_probability_label(victim_model, text)
    return prob_label == real_label

# PWWS
def get_saliency_score(victim_model, origin_text, transform_text, label):
    origin_probs = _get_probability(victim_model,origin_text)
    transform_probs = _get_probability(victim_model, transform_text)
    return caculate_saliency_score(origin_probs, transform_probs, label)

def caculate_saliency_score(origin_probs, transform_probs, label):
    score = origin_probs[label] - transform_probs[label]
    return score


# VALCAT
def get_round_score(origin_text, transform_text, true_label, attack_label):
    origin_probs = _get_probability(origin_text)
    transform_probs = _get_probability(transform_text)
    score = (origin_probs[true_label] - transform_probs[true_label])
    if not attack_label == true_label:
        score = score + (transform_probs[attack_label] - origin_probs[attack_label])
    return score


def _get_probability(victim_model, text):
    logits = victim_model(text)
    probs = softmax(logits)
    query_increase()
    return probs

def _get_probability_label(victim_model, text):
    logits = victim_model(text)
    probs = softmax(logits)
    prob_label = np.argmax(probs)
    query_increase()
    return prob_label

def __get_logits(victim_model, text):
    logits = victim_model(text)
    query_increase()
    return logits

def __get_logit_label(victim_model, text):
    logits = victim_model(text)
    logit_label = np.argmax(logits)
    query_increase()
    return logit_label



