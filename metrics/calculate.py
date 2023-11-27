import os
import numpy as np
import tensorflow_hub as hub
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_POOL
from pathlib import Path
from common import CalculationEntity, ATTACK_STATUS
from .persistence import Persistence



class MetricCalculator:

    def __init__(self):
        # USE_cache_path = os.path.join(Path(__file__).parent.parent.parent, MODEL_POOL['use_path'])
        # print("Calculator local_cache_path = {}".format(USE_cache_path))
        # self._use = hub.load(USE_cache_path)
        self._caculation_entity = None
        

    # 从文件加载
    def calculate_metrics():
        calculation_list = [] 
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

        print(f"adversarial attack result: success_attack_count = {success_count}, success_attack_rate = {success_attack_rate}, adv_perturbation_count = {ave_perturbation_count}, ave_query_count = {ave_query_count}")


def start_evaluation(origin_text, origin_label):
    calculator._caculation_entity = CalculationEntity()
    calculator._caculation_entity._attack_status = ATTACK_STATUS.FAILURE
    calculator._caculation_entity._origin_text = origin_text
    calculator._caculation_entity._origin_label = origin_label
def fresh_evaluation(adversary_text):
    calculator._caculation_entity._adversary_text = adversary_text
    return calculator._caculation_entity
    

def label(origin_label, attack_label):
    calculator._caculation_entity._origin_label = origin_label
    calculator._caculation_entity._attack_label = attack_label
def origin_label(label):
    calculator._caculation_entity._origin_label = label
def attack_label(label):
    calculator._caculation_entity._attack_label = label    


def query_increase():
    calculator._caculation_entity._query_count += 1

def attack_success():
    calculator._caculation_entity._attack_status = ATTACK_STATUS.SUCCESS

def attack_failure():
    calculator._caculation_entity._attack_status = ATTACK_STATUS.FAILURE

def perturbation_count(count: int):
    calculator._caculation_entity._perturbation_count = count

def append_origin_token(token):
    calculator._caculation_entity._origin_tokens.append(token)

def append_substituion_token(token):
    calculator._caculation_entity._substitution_tokens.append(token)    
def substituion_tokens(tokens):
    calculator._caculation_entity._substitution_tokens = tokens     



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


calculator = MetricCalculator()



