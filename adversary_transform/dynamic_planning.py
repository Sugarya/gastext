import numpy as np
from typing import Any
from utils import spacy_process
from metrics import calculate
from tqdm import tqdm
from victim_model import estimate


class DynamicPlanning:

    def __init__(self, victim_model):
        self._victim_model = victim_model


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        origin_text = args[0]
        candidate_lists = args[1]
        selection_unit_list = args[2]

        self.__greedy_search(candidate_lists, selection_unit_list, origin_text)

        # candidates_unit_tuple_list = self.__cluster(candidate_lists, selection_unit_list) 
        # need_search, result = self.__generate_graph(candidates_unit_tuple_list, origin_text)
        # if need_search:
        #     return self.__dynamic_search(result, origin_text)
        # else:
        #     return result
        
    '''
        脆弱值做聚类，输出第一组列表的大小
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
    def __generate_graph(self, candidates_unit_tuple_list, origin_text, attack_label = None):
        # Phase I: 确定图的连边
        origin_probs = estimate._get_probability(self._victim_model, origin_text)
        real_label = np.argmax(origin_probs)
        origin_token_list = spacy_process.tokenize(origin_text)
        # print(f"__generate_graph origin_logits = {origin_logits}")
        for candiates_unit_tuple in tqdm(candidates_unit_tuple_list, desc = 'replace a word and checkup'):
            origin_word = candiates_unit_tuple[1].word
            origin_position = candiates_unit_tuple[1].origin_position
            candidate_list = candiates_unit_tuple[0]
            for candidate in candidate_list:
                cp_origin_token_list = [*origin_token_list]
                # print(f"__generate_graph origin_word = {origin_word}, candidate.synonym = {candidate.synonym} replace {cp_origin_token_list[origin_position]}")
                cp_origin_token_list[origin_position] = candidate.synonym
                transform_text = spacy_process.detokenize(cp_origin_token_list)
                transform_probs = estimate._get_probability(self._victim_model, transform_text)
                transform_label = np.argmax(transform_probs)
                if not transform_label == real_label:
                    print("************************************************** FIND IT ****************************")
                    calculate.label(real_label, transform_label)
                    calculate.append_origin_token(origin_word)
                    calculate.perturbation_count(1)
                    calculate.attack_success()
                    calculate.append_substituion_token(candidate.synonym)
                    return False, transform_text
                
                candidate.fragile = estimate.caculate_saliency_score(origin_probs, transform_probs, real_label)
                candidate.transform_token_list = cp_origin_token_list

            candidate_list = list(filter(lambda t : t.fragile > 0, candidate_list))
            sorted_candidate_list = list(sorted(candidate_list, key = lambda t : t.fragile, reverse = True))
            # print(f"__generate_graph {len(candidate_list)} --> {len(sorted_candidate_list)}, sorted_candidate_list = {sorted_candidate_list}")
            candiates_unit_tuple[0] = sorted_candidate_list

        # Phase II: 构造图数据结构,candiates_unit_tuple_list
        candidates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, candidates_unit_tuple_list))
        for (sorted_candidate_list, unit) in tqdm(candidates_unit_tuple_list, desc = 'generate a graph'):
            unit.fragile = estimate.softmax(unit.saliency_score) * sorted_candidate_list[0].fragile
        sorted_candidates_unit_tuple_list = list(sorted(candidates_unit_tuple_list, key = lambda t : t[1].fragile, reverse = True))
        
        # print(f"__generate_graph sorted_candidates_unit_tuple_list = {sorted_candidates_unit_tuple_list}")
        return True, sorted_candidates_unit_tuple_list

    def __greedy_search(self, candidate_lists, selection_unit_list, origin_text):

        candidates_unit_tuple_list = list(filter(lambda t : len(t[0]) > 0, list(zip(candidate_lists, selection_unit_list))))
        candidates_unit_tuple_list = list(map(lambda t : [t[0], t[1]], candidates_unit_tuple_list))
        
        origin_probs = estimate._get_probability(self._victim_model, origin_text)
        real_label = np.argmax(origin_probs)
        origin_token_list = spacy_process.tokenize(origin_text)
        # print(f"__generate_graph origin_logits = {origin_logits}")
        for candiates_unit_tuple in tqdm(candidates_unit_tuple_list, desc = 'caculate saliency score'):
            origin_position = candiates_unit_tuple[1].origin_position
            candidate_list = candiates_unit_tuple[0]
            for candidate in candidate_list:
                cp_origin_token_list = [*origin_token_list]
                # print(f"__generate_graph origin_word = {origin_word}, candidate.synonym = {candidate.synonym} replace {cp_origin_token_list[origin_position]}")
                cp_origin_token_list[origin_position] = candidate.synonym
                transform_text = spacy_process.detokenize(cp_origin_token_list)
                transform_probs = estimate._get_probability(self._victim_model, transform_text)         
                candidate.fragile = estimate.caculate_saliency_score(origin_probs, transform_probs, real_label)

            # candidate_list = list(filter(lambda t : t.fragile > 0, candidate_list))
            sorted_candidate_list = list(sorted(candidate_list, key = lambda t : t.fragile, reverse = True))
            # print(f"__generate_graph {len(candidate_list)} --> {len(sorted_candidate_list)}, sorted_candidate_list = {sorted_candidate_list}")
            candiates_unit_tuple[0] = sorted_candidate_list

        # Phase II
        for (sorted_candidate_list, unit) in tqdm(candidates_unit_tuple_list, desc = 'generate a graph'):
            unit.fragile = estimate.softmax(unit.saliency_score) * sorted_candidate_list[0].fragile
        sorted_candidates_unit_tuple_list = list(sorted(candidates_unit_tuple_list, key = lambda t : t[1].fragile, reverse = True))

        # Phase III
        origin_token_list_backup = spacy_process.tokenize(origin_text)
        for i, candidates_unit in enumerate(sorted_candidates_unit_tuple_list):
            print(f"__greedy_search candidates_unit = {candidates_unit}")
            candidates, unit = candidates_unit[0], candidates_unit[1]
            origin_position = unit.origin_position
            origin_token_list_backup[origin_position] = candidates[0].synonym

            transform_text = spacy_process.detokenize(origin_token_list_backup)
            transform_probs = estimate._get_probability(self._victim_model, transform_text)
            max_transform_label = np.argmax(transform_probs)
            if not max_transform_label == real_label:
                print("**************************************************** Greedy FIND IT ****************************")
                calculate.label(real_label, transform_text)
                calculate.attack_success()
                return transform_text
            
            if i > 8:
                break

        return ''

    '''
        动态规划查找,借助transform_token_list, temp_fragile两个属性值
    '''
    def __dynamic_search(self, candidates_unit_tuple_list, origin_text, attack_label = None):
        origin_probs = estimate._get_probability(self._victim_model, origin_text)
        real_label = np.argmax(origin_probs)

        candiates_unit_len = len(candidates_unit_tuple_list)
        count = 0
        for i in range(1, candiates_unit_len):
            count = count + len(candidates_unit_tuple_list[i][0])
        with tqdm(total = count) as pbar:
            pbar.set_description('dynamic planing')
            for i in range(0, candiates_unit_len):
                # 统计指标
                calculate.append_origin_token(candidates_unit_tuple_list[i][1].word)
                calculate.perturbation_count(i + 1)
                if i == 0:
                    continue
                candidates_unit_tuple = candidates_unit_tuple_list[i]
                candidates = candidates_unit_tuple[0]
                unit = candidates_unit_tuple[1]
                origin_position = unit.origin_position
                previous_candidates = candidates_unit_tuple_list[i - 1][0]
                for _, candidate in enumerate(candidates):

                    for previous_candi in previous_candidates:
                        previous_candi.transform_token_list[origin_position] = candidate.synonym
                        transform_text = spacy_process.detokenize(previous_candi.transform_token_list)
                        transform_probs = estimate._get_probability(self._victim_model, transform_text)
                        max_transform_label = np.argmax(transform_probs)
                        if not max_transform_label == real_label:
                            print("**************************************************** DynamicPlan FIND IT ****************************")
                            calculate.label(real_label, transform_text)
                            calculate.attack_success()
                            return transform_text
                        previous_candi.temp_fragile = estimate.caculate_saliency_score(origin_probs, transform_probs, real_label)         

                    max_index = np.argmax(list(map(lambda t : t.temp_fragile, previous_candidates)))
                    candidate.transform_token_list = previous_candidates[max_index].transform_token_list
                    candidate.temp_fragile = previous_candidates[max_index].temp_fragile

                    pbar.update(1)


        return ''