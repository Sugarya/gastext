
import OpenHowNet
from utils import nlp_process
from common import BabelnetSubstitution


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
        

    '''
    pharse: 
        class OriginalPhrase:
        token = attr.ib()
        pos_tag = attr.ib()
        sentence_index = attr.ib()
        position_list = attr.ib() # [start_index, end_index]

    class BabelnetSubstitution:
        original_token = attr.ib() # ''
        candidate_tokens = attr.ib() # ['', ''], 同义词集
        position_list = attr.ib() # original_token在文本中的位置 [sentent_index, start_index, end_index]，
    '''    
    def generate_babelnet_substitution(self, origin_phrase_list):
        if not isinstance(origin_phrase_list, list):
            return []
        candicate_list = []
        # print(f"generate_babelnet_substitution len(origin_phrase_list) = {len(origin_phrase_list)}")
        for _, phrase in enumerate(origin_phrase_list):
            token, pos_tag = phrase.token.lower(), nlp_process.get_wordnet_pos(phrase.pos_tag)
            sentence_index, position_list = phrase.sentence_index, phrase.position_list
            syn_word_list = self._get_syn_words(token, pos_tag)
            candicate_list.append(BabelnetSubstitution(phrase, syn_word_list, [sentence_index, position_list[0], position_list[1]]))
        return candicate_list
    '''
        pos: Can be set to a/v/n/r
    '''
    def _get_syn_words(self, word, word_pos = None, language = LANGUAGE.EN):
        word_syn_list = []
        if self._hownet_dict_advanced.has(word, language):
            syn_list = self._hownet_dict_advanced.get_synset(word, pos = word_pos, language = language)
            # print(f"BabelnetConceptGenerator word = {word}, word_pos = {word_pos}, language = {language}, syn_list = {syn_list}")
            if language == LANGUAGE.EN: 
                for syn in syn_list:
                    word_syn_list = list(map(lambda s: format(s, word), syn.en_synonyms))
            else: 
                for syn in syn_list:
                    word_syn_list = list(map(lambda s: format(s, word), syn.zh_synonyms))
            print(f"BabelnetConceptGenerator get_syn_words word_syn_list = {word_syn_list}")
            return word_syn_list
        return []

def format(word, reference_word):
    word = remove_underline(word)
    return recover_word_case(word, reference_word)

def recover_word_case(word, reference_word):
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        return word
    
'''
[bn:00005054n|apple|苹果, bn:00005076n|apple_tree|苹果树]
'''    
def remove_underline(word):
    if not isinstance(word, str):
        return word
    elif '_' in word:
        return ''.join(word.split('_'))
    return word