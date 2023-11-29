
import math
import numpy as np
from typing import Any
from utils import spacy_process
from victim_model import estimate



'''
    划分器,并计算脆弱值
'''
class Separation:

    def __init__(self, victim_model):
        self._victim_model = victim_model


    def select_joint_unit(self, origin_text, real_label, split_unit_list):
        raw_token_list = spacy_process.tokenize(origin_text)
        origin_logits = self._victim_model(origin_text)
        label_unit_list = [ [] for _ in range(len(origin_logits)) ]
        # 脆弱的合力机制
        split_unit_len = len(split_unit_list)
        for i, unit in enumerate(split_unit_list):
            position = unit.origin_position
            cp_token_list = [*raw_token_list]
            # print(f"word,  {unit.word} =?= {cp_token_list[position]}")
            # del cp_token_list[position]
            cp_token_list[position] = 'unknown'
            transform_text = spacy_process.detokenize(cp_token_list)
            # print(f"Separation __call__ transform_text = {transform_text}")
            transform_logits = self._victim_model(transform_text)
            diff_logits = list(map(lambda t : abs(t[0] - t[1]), list(zip(transform_logits, origin_logits))))
            max_diff_label = np.argmax(diff_logits)
            unit.fragile_value = diff_logits[max_diff_label]
            label_unit_list[max_diff_label].append(unit)

            if split_unit_len <= 6:
                diff_logits[max_diff_label] = -1
                second_max_index = np.argmax(diff_logits)
                unit.secondary_fragile = diff_logits[second_max_index]
                label_unit_list[second_max_index].append(unit)

        counts = list(map(lambda t : len(t), label_unit_list))
        attack_label = np.argmax(counts)
        if attack_label == real_label:
            counts[attack_label] = -1
            attack_label = np.argmax(counts)
        # print(f"Separation __call__ len(split_unit_list) = {len(split_unit_list)}  count_list = {counts}, attack_label = {attack_label}, real_label= {real_label}")
        selection_label_unit_list = sorted(label_unit_list[attack_label], key = lambda t : t.fragile_value, reverse = True)
        # print(f"Separation __call__  result = {selection_unit_list}")

        return selection_label_unit_list, attack_label

    def caculate_saliency(self, *args: Any, **kwds: Any) -> Any:
        origin_text = args[0]
        real_label = args[1]
        split_unit_list = args[2]

        place_holder = 'unknown'
        raw_token_list = spacy_process.tokenize(origin_text)
        for i, unit in enumerate(split_unit_list):
            position = unit.origin_position
            cp_token_list = [*raw_token_list]
            cp_token_list[position] = place_holder
            transform_text = spacy_process.detokenize(cp_token_list)
            unit.saliency_score = estimate.get_saliency_score(self._victim_model, origin_text, transform_text, real_label)
            
        return split_unit_list
