
import OpenHowNet
from utils import nlp_process
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


    def generate_substitution(self, origin_unit_list):
        if not isinstance(origin_unit_list, list):
            return []
        candicate_list = []
        # print(f"generate_babelnet_substitution len(origin_phrase_list) = {len(origin_phrase_list)}")
        for _, unit in enumerate(origin_unit_list):
            net_pos_tag = nlp_process.get_wordnet_pos(unit.pos_tag)
            sentence_index, origin_position = unit.sentence_index, unit.origin_position
            syn_word_list = self._get_syn_words(unit.word, unit.lemma, net_pos_tag)
            candicate_list.append(NetSubstitution(unit.word, syn_word_list, sentence_index, origin_position))
        return candicate_list

    '''
        pos: Can be set to a/v/n/r
        使用单词原型,否则babelnet无法识别
    '''
    def _get_syn_words(self, word, lemma, word_pos = None, language = LANGUAGE.EN):
        word_syn_list = []
        if self._hownet_dict_advanced.has(lemma, language):
            syn_list = self._hownet_dict_advanced.get_synset(lemma, pos = word_pos, language = language)
            if language == LANGUAGE.EN: 
                for syn in syn_list:
                    word_syn_list = list(map(lambda s: nlp_process.format(s, word), syn.en_synonyms))
            else: 
                for syn in syn_list:
                    word_syn_list = list(map(lambda s: nlp_process.format(s, word), syn.zh_synonyms))
            print(f"BabelnetConceptGenerator word = {word}, word_pos = {word_pos}: get_syn_words word_syn_list = {word_syn_list}")
        return word_syn_list

