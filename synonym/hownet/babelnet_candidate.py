
import OpenHowNet
from utils import nlp_process


class LANGUAGE:
    ZH = 'zh'
    EN = 'en'

'''
    https://github.com/thunlp/OpenHowNet

'''
class BabelnetConceptGenerator:

    def __init__(self):
        OpenHowNet.download()
        self._hownet_dict_advanced = OpenHowNet.HowNetDict(init_babel = True)
        
        
    '''
        pos: Can be set to a/v/n/r
    '''
    def get_syn_words(self, word, word_pos = None, language = LANGUAGE.EN):
        if self._hownet_dict_advanced.has(word, language):
            word_pos = nlp_process.get_wordnet_pos(word_pos)
            print()
            syn_list = self._hownet_dict_advanced.get_synset(word, language = language, pos = word_pos)
        
            word_syn_list = []
            if language == LANGUAGE.EN: 
                for syn in syn_list:
                    en_synonyms = list(map(lambda t:recover_word_case(t, word),syn.en_synonyms))
                    word_syn_list.extend(en_synonyms)
            else: 
                for syn in syn_list:
                    zh_synonyms = list(map(lambda t: recover_word_case(t, word), syn.zh_synonyms))
                    word_syn_list.extend(zh_synonyms)
            return word_syn_list
        return []

def recover_word_case(word, reference_word):
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        return word
    

