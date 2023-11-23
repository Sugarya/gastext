

from typing import Any
from utils import spacy_process
import numpy as np

class DynamicPlanning:

    def __init__(self, victim_model):
        self._victim_model = victim_model


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        origin_text = args[0]
        candidate_lists = args[1]
        selection_unit_list = args[2]
        attack_label = kwds["attack"]

        candidates_unit_tuple_list = self.__cluster(candidate_lists, selection_unit_list) 
        need_search, result = self.__generate_graph(candidates_unit_tuple_list, attack_label, origin_text)
        if need_search:
            return self.__search(result, attack_label, origin_text)
        else:
            return result
    '''
        脆弱值做聚类，输出第一组列表的大小

        class Substitution:
            original_token 
            candidate_tokens
            sentence_index
            origin_postion
            fragile_value

        class OriginalUnit:
            word
            lemma
            pos_tag
            sentence_index
            origin_position
            spacy_token
            fragile_value
    '''
    def __cluster(self, candidate_lists, selection_unit_list):
        candidates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, list(zip(candidate_lists, selection_unit_list))))
        candidates_unit_tuple_list = list(map(lambda t : [t[0], t[1]], candidates_unit_tuple_list))
        # TODO 聚类

        if len(candidates_unit_tuple_list) >= 4:
            return candidates_unit_tuple_list[0:4]
        else:
            return candidates_unit_tuple_list
    '''

    '''
    def __generate_graph(self, candiates_unit_tuple_list, attack_label, origin_text):
        # Phase I: 确定图的连边
        origin_logits = self._victim_model(origin_text)
        real_label = np.argmax(origin_logits)
        origin_token_list = spacy_process.tokenize(origin_text)
        # print(f"__generate_graph origin_logits = {origin_logits}")
        for candiates_unit_tuple in candiates_unit_tuple_list:
            origin_word = candiates_unit_tuple[1].word
            origin_position = candiates_unit_tuple[1].origin_position
            candidate_list = candiates_unit_tuple[0]
            for candidate in candidate_list:
                cp_origin_token_list = [*origin_token_list]
                # print(f"__generate_graph origin_word = {origin_word}, candidate.synonym = {candidate.synonym} replace {cp_origin_token_list[origin_position]}")
                cp_origin_token_list[origin_position] = candidate.synonym
                transform_text = spacy_process.detokenize(cp_origin_token_list)
                transform_logits = self._victim_model(transform_text)
                # print(f"__generate_graph transform_logits = {transform_logits}")
                max_transform_label = np.argmax(transform_logits)
                # TODO 改成概率值最大,更新脆弱值计算方式
                if not max_transform_label == real_label:
                    print("************************************************** FIND IT ****************************")
                    print(f"__generate_graph change word count = 1, adversarial_example = {transform_text}")
                    print(f"__generate_graph max_transform_label = {max_transform_label}, real_label = {real_label}")
                    return False, transform_text
                # TODO abs(t[0] - t[1]) or t[0] - t[1] 二选一
                if not attack_label == real_label:
                    candidate.fragile_value = (transform_logits[attack_label] - origin_logits[attack_label]) + (origin_logits[real_label] - transform_logits[real_label])
                else:
                    candidate.fragile_value = transform_logits[attack_label] - origin_logits[attack_label]
                candidate.transform_token_list = cp_origin_token_list
                # max_label = np.argmax(diff_logits)
                # print(f"__generate_graph attack_label = {attack_label}, max_label = {max_label}  diff_logits = {diff_logits} ")
                # if diff_logits[attack_label] > 0 and max_label == attack_label:
                #     candidate.fragile_value = diff_logits[max_label]
                #     candidate.transform_token_list = cp_origin_token_list
            filter_candidate_list = list(filter(lambda t : t.fragile_value > 0, candidate_list))
            sorted_candidate_list = list(sorted(filter_candidate_list, key = lambda t : t.fragile_value, reverse = True))
            # print(f"__generate_graph {len(candidate_list)} --> {len(sorted_candidate_list)}, sorted_candidate_list = {sorted_candidate_list}")
            candiates_unit_tuple[0] = sorted_candidate_list

        # Phase II: 构造图数据结构,candiates_unit_tuple_list
        filter_candidates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, candiates_unit_tuple_list))

        return True, filter_candidates_unit_tuple_list

    '''
        动态规划查找
    '''
    def __search(self, candidates_unit_tuple_list, attack_label, origin_text):
        origin_logits = self._victim_model(origin_text)
        real_label = np.argmax(origin_logits)
        candiates_unit_len = len(candidates_unit_tuple_list)
        for i in range(1, candiates_unit_len):
            candidates_unit_tuple = candidates_unit_tuple_list[i]
            candidates = candidates_unit_tuple[0]
            unit = candidates_unit_tuple[1]
            origin_position = unit.origin_position
            previous_candidates = candidates_unit_tuple_list[i - 1][0]
            
            for _, candidate in enumerate(candidates):
                for previous_candi in previous_candidates:
                    previous_candi.transform_token_list[origin_position] = candidate.synonym
                    transform_text = spacy_process.detokenize(previous_candi.transform_token_list)
                    transform_logits = self._victim_model(transform_text)
                    max_transform_label = np.argmax(transform_logits)
                    if not max_transform_label == real_label:
                        print("**************************************************** DynamicPlan FIND IT ****************************")
                        print(f"__search adversarial_example = {transform_text}")
                        print(f"__search change word count = {i}, max_transform_label = {max_transform_label}, real_label = {real_label}")
                        return transform_text
                    
                    if not attack_label == real_label:
                        previous_candi.fragile_value = (transform_logits[attack_label] - origin_logits[attack_label]) + (origin_logits[real_label] - transform_logits[real_label])
                    else:
                        previous_candi.fragile_value = transform_logits[attack_label] - origin_logits[attack_label]

                max_index = np.argmax(list(map(lambda t : t.fragile_value, previous_candidates)))
                candidate.transform_token_list = previous_candidates[max_index].transform_token_list
                candidate.fragile_value = previous_candidates[max_index].fragile_value
        return ''