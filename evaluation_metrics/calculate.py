import os
import numpy as np
import tensorflow_hub as hub
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_POOL
from pathlib import Path
from common import Calculation


class MetricCalculator:

    def __init__(self):
        self._calculation_list = []

        local_cache_path = os.path.join(Path(__file__).parent.parent.parent, MODEL_POOL['use_path'])
        print("Calculator local_cache_path = {}".format(local_cache_path))
        self._use = hub.load(local_cache_path)
        self._reset()

    def _reset(self):
        self._origin_sentences = ''
        self._query_count = 0
        self._perturbation_count = 0
        self._attack_status = 0


calculator = MetricCalculator()

class ATTACK_STATUS:
    FAILURE = 'failure'
    SUCCESS = 'success'

def start_evaluation(origin_sentences):
    calculator._origin_sentences = origin_sentences

def fresh_evaluation(adversary_sentences):
    calculator._calculation_list.append(Calculation(calculator._origin_sentences, adversary_sentences, 
                calculator._query_count,  calculator._perturbation_count, calculator._attack_status))
    calculator._reset()

def query_increase():
    calculator._query_count += 1

def update_attack_status(status: str):
    calculator._attack_status = status

def update_perturbation_count(count: int):
    print("calculator update_perturbation_count = {}".format(count))
    calculator._perturbation_count = count

def get_calculation_list():
    result = [*calculator._calculation_list]
    calculator._calculation_list.clear()
    return result

# similary：USE + cosine distance
def get_use_sim(self, origin_example, adversarial_example):
    orig_embd, adv_embd = self._use([origin_example, adversarial_example]).numpy()
    sim = cosine_similarity(orig_embd[np.newaxis, ...], adv_embd[np.newaxis, ...])[0, 0]
    return sim.item()
    
'''
    origin_sentences = attr.ib()
    adversary_sentences = attr.ib()
    query_count = attr.ib()
    perturbation_count = attr.ib()
    attack_status = attr.ib() # 1 成功，0 失败
'''
def calculate_metrics(calculation_list):
    if not isinstance(calculation_list, list) or len(calculation_list) <= 0:
        return
    total_count = len(calculation_list)
    success_count = 0
    perturbation_count = 0
    query_count = 0
    for calulation in calculation_list:
        if calulation.attack_status == ATTACK_STATUS.SUCCESS:
            success_count += 1
            perturbation_count += calulation.perturbation_count
            query_count += calulation.query_count
    
    success_attack_rate = success_count / total_count
    ave_perturbation_count
    ave_query_count
    if success_count > 0:
        ave_perturbation_count = perturbation_count / success_count
        ave_query_count = query_count / success_count
    else:
        ave_perturbation_count = None
        ave_query_count = None

    print(f"adversarial attack result: success_attack_count = {success_count}, success_attack_rate = {success_attack_rate}, adv_perturbation_count = {ave_perturbation_count}, querave_query_county_count = {ave_query_count}")