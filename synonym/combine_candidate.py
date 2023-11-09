from typing import Any
from .wordnet.wordnet_candidate import WordnetCandidateGenerator
from .fillmask.encoder_decoder_candidate import FillMaskCandidateGenerator
from utils import tokenize

class SubstitutionListCombination:

    def __init__(self):
        self._wordnet_generator = WordnetCandidateGenerator()
        self._fill_mask_generator = FillMaskCandidateGenerator()
    
    def __call__(self, origin_phrase_list, origin_sentence_list):
        substitution_list = []
        # wordnet_substitution_list = self._wordnet_generator.generate_wordnet_substitution(origin_phrase_list)
        # print("__call__ wordnet_substitution_list = {}".format(wordnet_substitution_list))
        mask_substitution_list = self._fill_mask_generator.generate_mask_substitution(origin_phrase_list, origin_sentence_list)
        # print("__call__ mask_substitution_list = {}".format(mask_substitution_list))
        
        mask_substitution_list = self._flat_filter(origin_phrase_list, mask_substitution_list)
        return mask_substitution_list
        

    def _flat_filter(self, origin_sentence_list, mask_substitution_list):
        addition_size_list = []
        count = 0
        for _, sentence in enumerate(origin_sentence_list):
            addition_size_list.append(count)
            count += len(tokenize(sentence))

        for substitution in mask_substitution_list:
            position_list = substitution.position_list
            sentent_index = position_list[0]
            position_list[1] = position_list[1] + addition_size_list[sentent_index]
            position_list[2] = position_list[2] + addition_size_list[sentent_index]

        mask_substitution_randking = sorted(mask_substitution_list, key = lambda t: t.position_list[2], reverse = False)
        print("__call__ mask_substitution_randking = {}".format(mask_substitution_randking))

        pointer = 0
        next_pointer = pointer + 1
        filter_randking = []
        filter_randking.append(mask_substitution_randking[pointer])

        substitution_size = len(mask_substitution_randking)
        while next_pointer < substitution_size:
            if mask_substitution_randking[pointer].position_list[2] >= mask_substitution_randking[next_pointer].position_list[1]:
                next_pointer += 1
            else:
                filter_randking.append(mask_substitution_randking[next_pointer])
                pointer = next_pointer
                next_pointer = pointer + 1
        print("__call__ filter_randking = {}".format(filter_randking))
        return filter_randking