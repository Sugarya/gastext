from nltk.corpus import wordnet as wn
from lemminflect import getInflection
from enum import Enum

SUPPORTED_POS_TAGS = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]

def get_wordnet_pos(treebank_tag):
    treebank_tag = treebank_tag.upper()
    if treebank_tag.startswith('J') or treebank_tag.startswith('A'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''


def format_synonym(origin_word, lemma, pos_tag):
    # print(f"format_synonym origin_word = {origin_word}, lemma = {lemma}, pos_tag = {pos_tag}")
    inflection_word = transform_inflection(lemma, pos_tag)
    return _recover_word_case(inflection_word, origin_word)


def format(word, reference_word):
    word = _remove_underline(word)
    return _recover_word_case(word, reference_word)


'''
词的屈折变换
'''
def transform_inflection(lemma, pos_tag):
    transform_word = lemma
    if isinstance(lemma, str):
        if not '_' in lemma:
            transform_word = __get_inflection(lemma, pos_tag)
        else:
            word_list = lemma.split('_')
            end_postion = len(word_list) - 1
            if pos_tag.startswith('V'):
                word_list[0] = __get_inflection(word_list[0], pos_tag)
            else:
                word_list[end_postion] = __get_inflection(word_list[end_postion], pos_tag)
            transform_word = ' '.join(word_list)
    return transform_word

def __get_inflection(lemma_word, pos_tag):
    inflection = lemma_word
    inflection_tuple = getInflection(lemma_word, pos_tag)
    if isinstance(inflection_tuple, tuple) and len(inflection_tuple) >= 1:
        inflection = inflection_tuple[0]
    return inflection


'''
词的大小写变换
'''
def _recover_word_case(word, reference_word):
    # 保持大小写
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    
    # 首字母大写
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        return word
    
'''
[bn:00005054n|apple|苹果, bn:00005076n|apple_tree|苹果树]
'''    
def _remove_underline(word):
    if not isinstance(word, str):
        return word
    elif '_' in word:
        return ' '.join(word.split('_'))
    return word

def _pre_process_string(text):
    text = text.replace("%", "")
    text = text.replace(" '", "'")
    text = text.replace("$", "")
    return text


