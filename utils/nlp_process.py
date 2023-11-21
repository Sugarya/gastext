import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_sm')
treebank_word_detokenizer = TreebankWordDetokenizer()

def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def detokenize(tokens):
    return treebank_word_detokenizer.detokenize(tokens)


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
    
def format(word, reference_word):
    word = remove_underline(word)
    return recover_word_case(word, reference_word)

def recover_word_case(word, reference_word):
    # TODO 增加时态，语态
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
        return ' '.join(word.split('_'))
    return word

def _pre_process_string(text):
    text = text.replace("%", "")
    text = text.replace(" '", "'")
    text = text.replace("$", "")
    return text