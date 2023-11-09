from transformers import BertForMaskedLM, AutoTokenizer, BertConfig, pipeline
from config_attack import FILL_MASK_MODEL, DEVICES
from utils import Argument_Dict
from common import SubstitutionCandidate, OriginPhrase
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

nlp = spacy.load('en_core_web_sm')

class FillMaskCandidate:

    def __init__(self):
        encoder_decoder_type = Argument_Dict["encoder_decoder"]
        model_path , config_json = FILL_MASK_MODEL[encoder_decoder_type][0], FILL_MASK_MODEL[encoder_decoder_type][1]
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        model_config = BertConfig.from_json_file(config_json)
        self._model = BertForMaskedLM.from_pretrained(model_path, config=model_config).to(DEVICES[1])
        
        self._mask_token = '[MASK]'
        self._max_text_length = 512
        self._treebank_word_detokenizer = TreebankWordDetokenizer()


    def _encode_text(self, text):
        encoding = self._tokenizer(
            text,
            add_special_tokens = True,
            max_length=self._max_text_length,
            truncation = True,
            padding = 'max_length',
            return_tensors='pt',
        )
        return encoding.to(DEVICES[1])

    def _tokenize(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def _detokenize(self, tokens):
        return self._treebank_word_detokenizer.detokenize(tokens)

    def generate_mask_substitution(self, origin_phrase_list, origin_sentence_list):

        for index, origin_phrase in enumerate(origin_phrase_list):
            sentence_list = [*origin_sentence_list]
            mask_text = ''
            token, sentence_index, position_list = origin_phrase.token, origin_phrase.sentence_index, origin_phrase.position_list
            for i in range(len(sentence_list)):
                if i == sentence_index:
                    word_list = self._tokenize(sentence_list[i])
                    start, end = position_list[0], position_list[1] 
                    for j in range(start, end):
                        word_list[j] = ''
                    word_list[start] = self._mask_token
                    sentence_list[i] = self._detokenize(word_list)
            
            mask_text = ' '.join(sentence_list)
            print("generate_mask_substitution token = {}".format(token))
            
            unmasker = pipeline('fill-mask', model=self._model, tokenizer=self._tokenizer) 
            output = unmasker(mask_text)
            result_list = list(map(lambda element : element['token_str'], output))
            print("generate_mask_substitution result = {}".format(result_list))
            
        return SubstitutionCandidate(token, result_list)


