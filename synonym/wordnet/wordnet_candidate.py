
from functools import partial
from nltk.corpus import wordnet as wn
from .name_entity_list import NE_list
from common import WordnetSubstitution, OriginalPhrase


# https://www.section.io/engineering-education/getting-started-with-nltk-wordnet-in-python/


# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
supported_pos_tags = [
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

class WordnetCandidateGenerator:
    
    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return ''

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
            if pos_tag in supported_pos_tags:
                wordnet_post = self._get_wordnet_pos(pos_tag)
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

