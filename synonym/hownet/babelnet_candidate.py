
import OpenHowNet
from utils import nlp_process, spacy_process
from common import NetSubstitution


class LANGUAGE:
    ZH = 'zh'
    EN = 'en'

'''
    https://github.com/thunlp/OpenHowNet

'''
class BabelnetConceptGenerator:

    def __init__(self):
        # OpenHowNet.download()
        self._hownet_dict_advanced = OpenHowNet.HowNetDict()
        self._hownet_dict_advanced.initialize_babelnet_dict()


    def generate_substitution(self, origin_unit_list, similarity = 0.06):
        if not isinstance(origin_unit_list, list):
            return []
        candicate_list = []
        for _, unit in enumerate(origin_unit_list):
            origin_word, lemma, pos_tag = unit.word, unit.lemma, unit.pos_tag
            sentence_index, origin_position = unit.sentence_index, unit.origin_position
            syn_list = self._get_syn_words(lemma, pos_tag)
            filter_syn_list = spacy_process.filter_similar(unit.spacy_token, syn_list, similarity)
            format_syn_list = list(map(lambda s: nlp_process.format_synonym(origin_word, s, pos_tag), filter_syn_list))
            print(f"Babelnet generate_substitution origin_word = {origin_word}, lemma = {lemma}, pos_tag = {pos_tag}, format_syn_list = {format_syn_list}")
            candicate_list.append(NetSubstitution(unit.word, format_syn_list, sentence_index, origin_position))
        return candicate_list

    '''
        pos: Can be set to a/v/n/r
        使用单词原型,否则babelnet无法识别
    '''
    def _get_syn_words(self, lemma, word_pos, language = LANGUAGE.EN):
        total_syn_list = []
        if self._hownet_dict_advanced.has(lemma, language):
            word_pos = nlp_process.get_wordnet_pos(word_pos)
            syn_list = self._hownet_dict_advanced.get_synset(lemma, pos = word_pos, language = language)
            if language == LANGUAGE.EN:
                for syn in syn_list:
                    # print(f"_get_syn_words word_syn_list = {word_syn_list}")
                    total_syn_list.extend(syn.en_synonyms)
            else: 
                for syn in syn_list:
                    total_syn_list.extend(syn.zh_synonyms)
            # print(f"BabelnetConceptGenerator word = {word}, word_pos = {word_pos}: get_syn_words word_syn_list = {word_syn_list}")
        return list(set(total_syn_list))

