from typing import Any
from .wordnet.wordnet_candidate import WordnetCandidateGenerator
from .fillmask.encoder_decoder_candidate import FillMaskCandidateGenerator
from utils import tokenize
from common import Substitution


class SubstitutionListCombination:

    def __init__(self):
        self._wordnet_generator = WordnetCandidateGenerator()
        self._fill_mask_generator = FillMaskCandidateGenerator()
    
    def __call__(self, origin_phrase_list, origin_sentence_list):
        # wordnet_substitution_list = self._wordnet_generator.generate_wordnet_substitution(origin_phrase_list)
        # print("__call__ wordnet_substitution_list = {}".format(wordnet_substitution_list))
        mask_substitution_list = self._fill_mask_generator.generate_mask_substitution(origin_phrase_list, origin_sentence_list)
        # print("__call__ mask_substitution_list = {}".format(mask_substitution_list))
        
        substitution_list = self._flat_filter_map(origin_sentence_list, mask_substitution_list)
        return substitution_list
        
    # 过滤
    def _flat_filter_map(self, origin_sentence_list, mask_substitution_list):
        addition_list = []
        count = 0
        for _, sentence in enumerate(origin_sentence_list):
            sentence_length = len(tokenize(sentence))
            addition_list.append(count)
            count += sentence_length

        for substitution in mask_substitution_list:
            position_list = substitution.position_list
            sentent_index = position_list[0]
            position_list[1] = position_list[1] + addition_list[sentent_index]
            position_list[2] = position_list[2] + addition_list[sentent_index]
        mask_substitution_randking = sorted(mask_substitution_list, key = lambda t: t.position_list[2], reverse = False)        

        pointer = 0
        next_pointer = pointer + 1
        filter_substitution_randking = []
        filter_substitution_randking.append(mask_substitution_randking[0])
        # 贪心 活动安排
        substitution_size = len(mask_substitution_randking)
        while next_pointer < substitution_size:
            if mask_substitution_randking[pointer].position_list[2] >= mask_substitution_randking[next_pointer].position_list[1]:
                next_pointer += 1
            else:
                filter_substitution_randking.append(mask_substitution_randking[next_pointer])
                pointer = next_pointer
                next_pointer = pointer + 1
        print("_flat_filter_map filter_substitution_randking = {}".format(filter_substitution_randking))

        # 位置还原
        for substitution in filter_substitution_randking:
            position_list = substitution.position_list
            sentent_index = position_list[0]
            position_list[1] = position_list[1] - addition_list[sentent_index]
            position_list[2] = position_list[2] - addition_list[sentent_index]

        result = list(map(
            lambda t : Substitution(
                t.original_token, 
                t.candidate_tokens, 
                t.position_list,
                [*origin_sentence_list]),
            filter_substitution_randking))
        print("_flat_filter_map result = {}".format(result))

        return result