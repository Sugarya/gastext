
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
        origin_text = args[0]
        real_label = args[1]

        raw_token_list = spacy_process.tokenize(origin_text)
        origin_logits = self._victim_model(origin_text)
        if not real_label == np.argmax(origin_logits):
            print("********************Error**********, real_label {real_label} != argmax index")
        label_unit_list = [ [] for _ in range(len(origin_logits)) ]

        split_unit_list = spacy_process.split(origin_text)
        for i, unit in enumerate(split_unit_list):
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

            max_label = np.argmax(diff_logits)
            max_fragile = diff_logits[max_label]
            unit.fragile_value = max_fragile
            label_unit_list[max_label].append(unit)
            # print(f"Separation __call__ max_index = {max_label}, i = {i}, unit.fragile_value = {unit.fragile_value}")

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
        selection_label_unit_list = sorted(label_unit_list[attack_label], key = lambda t : t.fragile_value, reverse = True)
        # print(f"Separation __call__  result = {selection_unit_list}")
        return selection_label_unit_list, attack_label
