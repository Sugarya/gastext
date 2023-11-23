

from typing import Any
from utils import spacy_process
import numpy as np

class DynamicPlanning:

    def __init__(self, victim_model) -> None:
        self._victim_model = victim_model
        self._transform_text = ''


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        origin_text = args[0]
        candidate_lists = args[1]
        selection_unit_list = args[2]
        attack_label = kwds["attack"]

        candiates_unit_tuple_list = self.__cluster(candidate_lists, selection_unit_list) 
        enable = self.__generate_graph(candiates_unit_tuple_list, attack_label, origin_text)

        if enable:
            pass
        
        return self._transform_text
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
        candiates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, list(zip(candidate_lists, selection_unit_list))))
        candiates_unit_tuple_list = list(map(lambda t : [t[0], t[1]], candiates_unit_tuple_list))
        # TODO

        if len(candiates_unit_tuple_list) >= 4:
            return candiates_unit_tuple_list[0:4]
        else:
            return candiates_unit_tuple_list
    '''

    '''
    def __generate_graph(self, candiates_unit_tuple_list, attack_label, origin_text):
        # Phase I: 确定图的连边
        origin_logits = self._victim_model(origin_text)
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
                # TODO 改成概率值最大
                if max_transform_label == attack_label:
                    print("**************************************************FIND IT****************************")
                    self._transform_text = transform_text
                    return False
                # TODO abs(t[0] - t[1]) or t[0] - t[1] 二选一
                diff_logits = list(map(lambda t : t[0] - t[1], list(zip(transform_logits, origin_logits))))
                # TODO 二选一
                if diff_logits[attack_label] > 0:
                    candidate.fragile_value = diff_logits[attack_label]
                # max_label = np.argmax(diff_logits)
                # print(f"__generate_graph attack_label = {attack_label}, max_label = {max_label}  diff_logits = {diff_logits} ")
                # if diff_logits[attack_label] > 0 and max_label == attack_label:
                #     candidate.fragile_value = diff_logits[max_label]
            filter_candidate_list = list(filter(lambda t : t.fragile_value > 0, candidate_list))
            sorted_candidate_list = list(sorted(filter_candidate_list, key = lambda t : t.fragile_value, reverse = True))
            print(f"__generate_graph {len(candidate_list)} --> {len(sorted_candidate_list)}, sorted_candidate_list = {sorted_candidate_list}")
            candiates_unit_tuple[0] = sorted_candidate_list

        # Phase II: 构造图数据结构,candiates_unit_tuple_list
        filter_candiates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, candiates_unit_tuple_list))
        

        return True

    def __search(self):

        
        pass    

