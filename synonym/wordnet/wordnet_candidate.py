
from nltk.corpus import wordnet as wn
from common import NetSubstitution
from utils import nlp_process, spacy_process

# https://www.section.io/engineering-education/getting-started-with-nltk-wordnet-in-python/
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


class WordnetCandidateGenerator:
    

    def generate_substitution_by_phrases(self, origin_phrase_list):
        # TODO 增加识别NE
        # NE_candidates = NE_list.L[dataset_name][true_label]
        substitution_list = []
        for origin_phrase in origin_phrase_list:
            phrase_text, pos_tag = nlp_process._pre_process_string(origin_phrase.token), origin_phrase.pos_tag
            sentence_index, position_list = origin_phrase.sentence_index, origin_phrase.position_list
            synonym_list = self._word_candidate(phrase_text, pos_tag)
            substitution = NetSubstitution(phrase_text, synonym_list, [sentence_index, position_list[0], position_list[1]])  
            substitution_list.append(substitution)
        return substitution_list

    def generate_substitution(self, origin_unit_list, similarity = 0.06):
        substitution_list = []
        for _, unit in enumerate(origin_unit_list):
            origin_word, lemma, pos_tag = unit.word, unit.lemma, unit.pos_tag
            synonym_list = self._word_candidate(lemma, pos_tag)
            filter_synonym_list = spacy_process.filter_similar(unit.spacy_token, synonym_list, similarity)
            format_synonym_list = list(map(lambda t : nlp_process.format_synonym(origin_word, t, pos_tag) , filter_synonym_list))
            # print(f"Wordnet generate_substitution origin_word = {origin_word} lemma = {lemma}, pos_tag = {pos_tag}, format_synonym_list = {format_synonym_list}")
            substitution = NetSubstitution(origin_word, format_synonym_list, unit.origin_position)
            # print(f"generate_substitution_by_words substitution = {substitution}")
            substitution_list.append(substitution)
        return substitution_list


    def generate_substitution_by_words_list(self, words_list):
        substitution_list = []
        for i, words in enumerate(words_list):
            for j, word in enumerate(words):
                synonym_list = self._word_candidate(word)
                substitution = NetSubstitution(word, synonym_list, i, j)
                print(f"_word_candidate substitution = {substitution}")
                substitution_list.append(substitution)
        return substitution_list
    
    def _word_candidate(self, word, pos_tag = ''):
        synsets = []
        try:
            wordnet_pos = nlp_process.get_wordnet_pos(pos_tag)
            synsets = wn.synsets(word, pos = wordnet_pos)
        except (RuntimeError, KeyError):
            synsets = []
        wordnet_synonyms = [ synset.lemma_names() for synset in synsets]
        synonym_list = []
        for synonym in wordnet_synonyms:
            synonym_list.extend(synonym)
        synonym_list = list(set(synonym_list))
        return synonym_list

    
