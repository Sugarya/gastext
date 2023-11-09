import attr
import spacy
from functools import partial
from nltk.corpus import wordnet
from .name_entity_list import NE_list

nlp = spacy.load('en_core_web_sm')
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# https://www.section.io/engineering-education/getting-started-with-nltk-wordnet-in-python/

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


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()

def generate_wordnet_synonym(token, position, dataset, true_y):
    NE_candidates = NE_list.L[dataset][true_y]
    NE_tags = list(NE_candidates.keys())        
    candidates = []
    NER_tag = token.ent_type_
    if NER_tag in NE_tags:
        candidate = SubstitutionCandidate(position, 0, token, NE_candidates[NER_tag])
        candidates.append(candidate)
    else:
        candidates = _generate_synonym_candidates(token, position)
    return candidates


def _generate_synonym_candidates(token, token_position, rank_fn = None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity
    
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        
        wordnet_synonyms = []
        synsets = wordnet.synsets(token.text, pos = wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)
        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidate = SubstitutionCandidate(
                token_position=token_position,
                similarity_rank=None,
                original_token=token,
                candidate_word=candidate_word)
            candidates.append(candidate)
    return candidates

def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)

def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj

def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True