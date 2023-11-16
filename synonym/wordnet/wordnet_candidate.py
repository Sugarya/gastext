
from functools import partial
from nltk.corpus import wordnet as wn
from .name_entity_list import NE_list
from common import WordnetSubstitution
from utils import nlp_process

# https://www.section.io/engineering-education/getting-started-with-nltk-wordnet-in-python/
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


class WordnetCandidateGenerator:
    
    def _process_string(self, text):
        text = text.replace("%", "")
        text = text.replace(" '", "'")
        text = text.replace("$", "")
        return text

    def _generate_synonym_candidate(self, origin_phrase):
        phrase_text, pos_tag = self._process_string(origin_phrase.token), origin_phrase.pos_tag
        print("_generate_synonym_candidates phrase_text = {}".format(phrase_text))
        
        synsets = []
        # synsets = wn.synsets(phrase_text, check_exceptions=False)
        try:
            if pos_tag in nlp_process.SUPPORTED_POS_TAGS:
                wordnet_post = nlp_process.get_wordnet_pos(pos_tag)
                print("_generate_synonym_candidates wordnet_post = {}".format(wordnet_post))
                synsets = wn.synsets(phrase_text, pos = wordnet_post)
            else:
                synsets = wn.synsets(phrase_text)    
        except (RuntimeError, KeyError):
            pass
        wordnet_synonyms = [ synset.lemma_names() for synset in synsets] # lemma_names() / lemmas
        synonym_list = []
        for synonym in wordnet_synonyms:
            synonym_list.extend(synonym)
        synonym_list = list(set(synonym_list))
        # print("_generate_synonym_candidates synonym_list = {}".format(synonym_list))
        sentence_index, position_list = origin_phrase.sentence_index, origin_phrase.position_list
        return WordnetSubstitution(phrase_text, synonym_list, [sentence_index, position_list[0], position_list[1]])  


    def generate_wordnet_substitution(self, origin_phrase_list):
        # TODO 增加识别NE
        # NE_candidates = NE_list.L[dataset_name][true_label]
        substitution_list = []
        for origin_phrase in origin_phrase_list:
            substitution = self._generate_synonym_candidate(origin_phrase)
            substitution_list.append(substitution)
        return substitution_list

