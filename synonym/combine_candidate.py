from typing import Any
from .wordnet.wordnet_candidate import WordnetCandidateGenerator
from .fillmask.encoder_decoder_candidate import FillMaskCandidateGenerator
from .hownet.babelnet_candidate import BabelnetConceptGenerator
from utils import tokenize
from common import Substitution


class SubstitutionListCombination:

    def __init__(self):
        self._wordnet_generator = WordnetCandidateGenerator()
        self._fill_mask_generator = FillMaskCandidateGenerator()
        self._babelnet_generator = BabelnetConceptGenerator()
    
    def __call__(self, origin_phrase_list, origin_sentence_list):
        print(f"__call__ origin_phrase_list = {origin_phrase_list}")
        # wordnet_substitution_list = self._wordnet_generator.generate_wordnet_substitution(origin_phrase_list)
        # print("__call__ wordnet_substitution_list = {}".format(wordnet_substitution_list))
        # mask_substitution_list = self._fill_mask_generator.generate_mask_substitution(origin_phrase_list, origin_sentence_list)
        # print("__call__ mask_substitution_list = {}".format(mask_substitution_list))

        origin_phrase_list = self._pre_flat_filter_map(origin_sentence_list, origin_phrase_list)
        babelnet_substitution_list = self._babelnet_generator.generate_babelnet_substitution(origin_phrase_list)
        # print(f"__call__ babelnet_substitution_list = {babelnet_substitution_list}")
        substitution_list = self._merge(babelnet_substitution_list)
        print(f"__call__ substitution_list = {substitution_list}")
        return substitution_list
        
    def _merge(self, *var_args):
        babelnet_substitution_list = var_args[0]
        substitutions = list(map(
            lambda t : Substitution(
                t.original_token, 
                t.candidate_tokens, 
                t.position_list),
            babelnet_substitution_list))
        print("_flat_filter_map result = {}".format(substitutions))
        return substitutions

    '''
        pharse: 
        class OriginalPhrase:
        token = attr.ib()
        pos_tag = attr.ib()
        sentence_index = attr.ib()
        position_list = attr.ib() # [start_index, end_index]
        
        短语存在位置重叠，按最早结束和最短长度词语做贪心筛选
    '''
    def _pre_flat_filter_map(self, origin_sentence_list, origin_phrase_list):
        if not isinstance(origin_phrase_list, list) or len (origin_phrase_list) <= 0:
            return []
        addition_list = []
        count = 0
        for _, sentence in enumerate(origin_sentence_list):
            sentence_length = len(tokenize(sentence))
            addition_list.append(count)
            count += sentence_length

        for phrase in origin_phrase_list:
            sentent_index = phrase.sentence_index
            position_list = phrase.position_list
            position_list[0] = position_list[0] + addition_list[sentent_index]
            position_list[1] = position_list[1] + addition_list[sentent_index]
        origin_phrase_randking = sorted(origin_phrase_list, key = lambda t: t.position_list[1], reverse = False)        
        print("_flat_filter_map origin_phrase_randking = {}".format(origin_phrase_randking))
        pointer = 0
        next_pointer = pointer + 1
        filter_phrase_randking = []
        filter_phrase_randking.append(origin_phrase_randking[0])
        # 贪心 活动安排
        substitution_size = len(origin_phrase_randking)
        while next_pointer < substitution_size:
            if origin_phrase_randking[pointer].position_list[1] >= origin_phrase_randking[next_pointer].position_list[0]:
                next_pointer += 1
            else:
                filter_phrase_randking.append(origin_phrase_randking[next_pointer])
                pointer = next_pointer
                next_pointer = pointer + 1
        print("_flat_filter_map filter_phrase_randking = {}".format(filter_phrase_randking))
        # 位置还原
        for phrase in filter_phrase_randking:
            sentent_index = phrase.sentence_index
            position_list = phrase.position_list
            position_list[0] = position_list[0] - addition_list[sentent_index]
            position_list[1] = position_list[1] - addition_list[sentent_index]

        return filter_phrase_randking