import spacy
from common import OriginalUnit


FILTER_POS_TAG = ['NUM', 'PUNCT', 'AUX']



'''
    提供：分词, pos, 命名实体识别
'''
class SpacyProcessor:

    def __init__(self) -> None:
        self._nlp = spacy.load('en_core_web_sm')
        # self._nlp = spacy.load('zh_core_web_sm')



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
    def print_tag_description(self):
        tag_descriptions = { tag: spacy.explain(tag) for tag in self._nlp.get_pipe('tagger').labels}
        for tag, description in tag_descriptions.items():
            print(f"{tag}: {description}")

    def tokenize(self, text):
        doc = self._nlp(text)
        name_entity_list = []
        for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            text_list = ent.text.split(' ')
            name_entity_list.extend(text_list)
        # print(f"SpacyProcessor tokenize name_entity_list = {name_entity_list}")
        
        token_list = []
        origin_unit_list = []
        for j, token in enumerate(doc):
            token_list.append(token)
            enable = not token.is_stop and not token.text in name_entity_list and not token.pos_ in FILTER_POS_TAG
            if enable:
                # print(f"SpacyProcessor tokenize: {token.text, token.lemma_, token.pos_, token.is_stop}")
                origin_unit_list.append(OriginalUnit(token.text, token.lemma_, token.pos_, 0, j))
        return origin_unit_list, token_list
    
    '''
        推断数据集，有多个文本
    '''
    def tokenize_list(self, texts):
        ne_list = []
        docs = [self._nlp(text) for text in texts]
        for doc in docs:
            for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                ne_list.append(ent.text)
        print(f"SpacyProcessor tokenize ne_list = {ne_list}")
        
        token_list = []
        for i, doc in enumerate(docs):
            for j, token in enumerate(doc):
                enable = not token.is_stop and not token.text in ne_list
                if enable:
                    print(f"SpacyProcessor tokenize: {token.text, token.lemma_, token.pos_, token.is_stop}")
                    token_list.append(OriginalUnit(token.text, token.lemma_, token.pos_, i, j))

        return token_list

            
         


