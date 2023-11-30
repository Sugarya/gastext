
import math
import numpy as np
from tqdm import tqdm
from typing import Any
from utils import spacy_process
from victim_model import estimate



'''
    计算脆弱值，划分器
'''
class Separation:

    def __init__(self, victim_model):
        self._victim_model = victim_model


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        candidate_lists, origin_unit_list, text, real_label = args[0], args[1],args[2],args[3]
        unit_candidates_list = self.__pattern_input(origin_unit_list, candidate_lists)
        unit_candidates_list, probs_len = self.__caculate_fragile(unit_candidates_list, text, real_label)
        result = self.__pattern_output(unit_candidates_list, probs_len)
        return result

    def __caculate_fragile(self, unit_candidates_list, origin_text, real_label):
        raw_token_list = spacy_process.tokenize(origin_text)
        origin_probs = estimate._get_probability(self._victim_model, origin_text)
        probs_len = len(origin_probs)

        tqdm_count = 0
        for _, (_, candidates) in enumerate(unit_candidates_list):
            tqdm_count = tqdm_count + len(candidates)
        with tqdm(total = tqdm_count, desc = 'caculate synonym fragile') as pbar:
            for i, (unit, candidates) in enumerate(unit_candidates_list):
                cp_token_list = [*raw_token_list]
                position = unit.origin_position
                for j , candidate in enumerate(candidates):
                    cp_token_list[position] = candidate.synonym
                    perturb_text = spacy_process.detokenize(cp_token_list)
                    perturb_probs = estimate._get_probability(perturb_text)
                    diff_probs = [ perturb_probs[i] - origin_probs[i] for i in range(probs_len)]
                    
                    real_label_fragile = -1 * diff_probs[real_label]
                    diff_probs[real_label] = real_label_fragile
                    candidate.diff_probs = diff_probs
                    fragile_probs = list(map(lambda t : t + real_label_fragile , diff_probs))
                    fragile_probs[real_label] = real_label_fragile
                    candidate.fragile_probs = fragile_probs    

                # 求和得 原始词等级分数
                for k in range(probs_len):
                    fragile_probs_list = list(map(lambda t : t.fragile_probs, candidates))
                    prob_list = list(map(lambda t : t[k]), fragile_probs_list)
                    unit.saliencies[k] = sum(list(filter(lambda t : t > 0, prob_list)))
                    
                pbar.update(1)

        return unit_candidates_list, probs_len

    '''
        规整输入的参数, 服务下游任务
    '''
    def __pattern_input(self, origin_unit_list, candidate_lists):
        unit_candidates_list = list(filter(lambda t : len(t[0]) > 0, list(zip(origin_unit_list, candidate_lists))))
        # unit_candidates_list = list(map(lambda t : [t[0], t[1]], unit_candidates_list))
        return unit_candidates_list

    def __pattern_output(self, unit_candidates_list, saliency_size):
        for i in range(saliency_size):
            pass


        
          

    # 脆弱的合力机制v1.0
    def __select_joint_unit(self, origin_text, real_label, split_unit_list):
        raw_token_list = spacy_process.tokenize(origin_text)
        origin_logits = self._victim_model(origin_text)
        label_unit_list = [ [] for _ in range(len(origin_logits)) ]
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

    def __caculate_saliency(self, *args: Any, **kwds: Any) -> Any:
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
