
import math
from typing import Any
from utils import spacy_process
import numpy as np

'''
    划分器,并计算脆弱值
'''
class Separation:

    def __init__(self, victim_model) -> None:
        self._victim_model = victim_model

    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        real_label = args[0]
        origin_text = args[1]
        origin_unit_list = kwds["unit_list"]

        origin_logits = self._victim_model(origin_text)
        # print(f"Separation __call__ origin_unit_list size = {len(origin_unit_list)}, origin logits = {origin_logits}")
        if not real_label == np.argmax(origin_logits):
            print("********************Error**********, real_label {real_label} != argmax index")
        
        raw_token_list = spacy_process.tokenize(origin_text)
        label_unit_list = [ [] for _ in range(len(origin_logits)) ]
        for i, unit in enumerate(origin_unit_list):
            position = unit.origin_position
            cp_token_list = [*raw_token_list]
            # print(f"word,  {unit.word} =?= {cp_token_list[position]}")
            # del cp_token_list[position]
            cp_token_list[position] = '[Unk]'
            transform_text = spacy_process.detokenize(cp_token_list)
            # print(f"Separation __call__ transform_text = {transform_text}")

            transform_logits = self._victim_model(transform_text)
            # print(f"Separation __call__ transform_logits = {transform_logits}")
            diff_logits = list(map(lambda t : abs(t[0] - t[1]), list(zip(transform_logits, origin_logits))))
            # print(f"Separation __call__ diff_logits = {diff_logits}")

            max_index = np.argmax(diff_logits)
            max_fragile = diff_logits[max_index]
            # print(f"Separation __call__ max_index = {max_index}, i = {i}, max_fragile = {max_fragile}")
            label_unit_list[max_index].append((unit, max_fragile))

            # diff_logits[max_index] = -1
            # print(f"Separation __call__ diff_logits after = {diff_logits}")
            # second_max_index = np.argmax(diff_logits)
            # second_max_fragile = diff_logits[second_max_index]
            # print(f"Separation __call__ second_max_index = {second_max_index}, i = {i}, second_max_fragile = {second_max_fragile}")
            # label_unit_list[second_max_index].append((unit, second_max_fragile))

        counts = list(map(lambda t : len(t), label_unit_list))
        attack_label = np.argmax(counts)
        if attack_label == real_label:
            counts[attack_label] = -1
            attack_label = np.argmax(counts)
        print(f"Separation __call__ counts = {counts}, attack_label = {attack_label}, real_label= {real_label}")
        selection_label_unit_list = sorted(label_unit_list[attack_label], key = lambda t : t[1], reverse = True)
        selection_unit_list = list(map(lambda t : t[0], selection_label_unit_list))
        selection_fragile_list = list(map(lambda t : t[1], selection_label_unit_list))
        # print(f"Separation __call__  result = {selection_unit_list}")
        return selection_unit_list, selection_fragile_list, attack_label
