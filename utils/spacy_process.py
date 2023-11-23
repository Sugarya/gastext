import spacy
import re
from common import OriginalUnit
from nltk.tokenize.treebank import TreebankWordDetokenizer

treebank_word_detokenizer = TreebankWordDetokenizer()
# spacy.prefer_gpu()
# TODO 更换更大的语言模型
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('zh_core_web_sm')

FILTER_POS_TAG = ['NUM', 'PUNCT', 'AUX']

'''
    词性标注 (TAG) 及其描述：
        $: symbol, currency
        '': closing quotation mark
        ,: punctuation mark, comma
        -LRB-: left round bracket
        -RRB-: right round bracket
        .: punctuation mark, sentence closer
        :: punctuation mark, colon or ellipsis
        ADD: email
        AFX: affix
        CC: conjunction, coordinating
        CD: cardinal number
        DT: determiner
        EX: existential there
        FW: foreign word
        HYPH: punctuation mark, hyphen
        IN: conjunction, subordinating or preposition
        JJ: adjective (English), other noun-modifier (Chinese)
        JJR: adjective, comparative
        JJS: adjective, superlative
        LS: list item marker
        MD: verb, modal auxiliary
        NFP: superfluous punctuation
        NN: noun, singular or mass
        NNP: noun, proper singular
        NNPS: noun, proper plural
        NNS: noun, plural
        PDT: predeterminer
        POS: possessive ending
        PRP: pronoun, personal
        PRP$: pronoun, possessive
        RB: adverb
        RBR: adverb, comparative
        RBS: adverb, superlative
        RP: adverb, particle
        SYM: symbol
        TO: infinitival "to"
        UH: interjection
        VB: verb, base form
        VBD: verb, past tense
        VBG: verb, gerund or present participle
        VBN: verb, past participle
        VBP: verb, non-3rd person singular present
        VBZ: verb, 3rd person singular present
        WDT: wh-determiner
        WP: wh-pronoun, personal
        WP$: wh-pronoun, possessive
        WRB: wh-adverb
        XX: unknown
        ``: opening quotation mark
'''
def print_tag_description():
    tag_descriptions = { tag: spacy.explain(tag) for tag in nlp.get_pipe('tagger').labels}
    for tag, description in tag_descriptions.items():
        print(f"{tag}: {description}")

def split(text):
    doc = nlp(text)
    name_entity_list = []
    for ent in doc.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        text_list = ent.text.split(' ')
        name_entity_list.extend(text_list)
    # print(f"SpacyProcessor tokenize name_entity_list = {name_entity_list}")
    
    origin_unit_list = []
    for j, token in enumerate(doc):
        enable = not token.is_stop and not token.text in name_entity_list and not token.pos_ in FILTER_POS_TAG
        if enable:
            # print(f"SpacyProcessor tokenize: {token.text, token.lemma_, token.tag_, token.is_stop}")
            origin_unit_list.append(OriginalUnit(token.text, token.lemma_, token.tag_, 0, j, token, 0))
    return origin_unit_list
    
'''
    推断数据集，有多个文本
'''
def split_list(texts):
    name_entity_list = []
    docs = [nlp(text) for text in texts]
    for doc in docs:
        for ent in doc.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            name_entity_list.append(ent.text)
    print(f"SpacyProcessor tokenize ne_list = {name_entity_list}")
    
    token_list = []
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            enable = not token.is_stop and not token.text in name_entity_list and not token.pos_ in FILTER_POS_TAG
            if enable:
                print(f"SpacyProcessor tokenize: {token.text, token.lemma_, token.pos_, token.is_stop}")
                token_list.append(OriginalUnit(token.text, token.lemma_, token.pos_, i, j, token, 0))

    return token_list

def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def detokenize(token_list):
    text = treebank_word_detokenizer.detokenize(token_list)
    text = re.sub('\s*,\s*', ', ', text)
    text = re.sub('\s*\.\s*', '. ', text)
    text = re.sub('\s*\?\s*', '? ', text)
    return text

def filter_similar(spacy_token, synonym_list, similary_threshold):
    sentence = ' '.join(synonym_list)
    origin_score_list = __get_similar_token_score(spacy_token, sentence)
    filter_tuple_list = list(filter(lambda t : t[0] > similary_threshold , list(zip(origin_score_list, synonym_list))))
    synonym_list = list(map(lambda t : t[1], filter_tuple_list))
    return synonym_list

'''
    筛选合并的同义词集, 在句子粒度的相似性
'''
def filter_similar_doc(origin_text, origin_position, candidate_synonyms, similary_threshold):
    score_list = []
    origin_doc = nlp(origin_text)
    origin_token_list = tokenize(origin_text)
    for synonym in candidate_synonyms:
        cp_origin_token_list = [*origin_token_list]
        # print(f"origin_token {cp_origin_token_list[origin_position]} =?= {original_token}")
        cp_origin_token_list[origin_position] = synonym
        synonym_text =  detokenize(cp_origin_token_list)
        synonym_doc = nlp(synonym_text)
        sim_score = origin_doc.similarity(synonym_doc)
        score_list.append(sim_score)
    zip_list = list(zip(score_list, candidate_synonyms))
    # print(f"filter_similar_doc zip_list = {zip_list}")
    filter_zip_list = filter(lambda t : t[0] > similary_threshold, zip_list)
    sorted_zip_list = sorted(filter_zip_list, key = lambda t : t[0], reverse = True)
    # print(f"filter_similar_doc sorted_zip_list = {sorted_zip_list}")
    return list(map(lambda t : t[1], sorted_zip_list))

def __get_similar_token_score(spacy_token, sentence):
    score_list = []
    doc = nlp(sentence)
    for token in doc:
        sim_score = token.similarity(spacy_token)
        score_list.append(sim_score)
    return score_list
