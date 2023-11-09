from transformers import BertForMaskedLM, AutoTokenizer, BertConfig, pipeline
from config_attack import FILL_MASK_MODEL, DEVICES
from utils import Argument_Dict
from common import SubstitutionCandidate, OriginalPhrase
from utils import detokenize, tokenize


class FillMaskCandidateGenerator:

    def __init__(self):
        encoder_decoder_type = Argument_Dict["encoder_decoder"]
        model_path , config_json = FILL_MASK_MODEL[encoder_decoder_type][0], FILL_MASK_MODEL[encoder_decoder_type][1]
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        model_config = BertConfig.from_json_file(config_json)
        self._model = BertForMaskedLM.from_pretrained(model_path, config=model_config).to(DEVICES[1])
        
        self._mask_token = '[MASK]'
        self._max_text_length = 512
        


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

    def generate_mask_substitution(self, origin_phrase_list, origin_sentence_list):
        substitution_list = []
        
        for _, origin_phrase in enumerate(origin_phrase_list):
            sentence_list = [*origin_sentence_list]
            mask_text = ''
            token, sentence_index, position_list = origin_phrase.token, origin_phrase.sentence_index, origin_phrase.position_list
            for i in range(len(sentence_list)):
                if i == sentence_index:
                    word_list = tokenize(sentence_list[i])
                    word_list_size = len(word_list)
                    
                    start, end = position_list[0], position_list[1]
                    print("generate_mask_substitution word_list = {}".format(word_list))
                    print("generate_mask_substitution start = {}, end = {}".format(start, end))
                    word_list[start] = self._mask_token
                    for j in range(start + 1, end + 1):
                        if j < word_list_size:
                            word_list[j] = ''
                    sentence_list[i] = detokenize(word_list)
            
            mask_text = ' '.join(sentence_list)
            print("generate_mask_substitution mask_text = {}".format(mask_text))
            print("generate_mask_substitution token = {}".format(token))
            
            unmasker = pipeline('fill-mask', model=self._model, tokenizer=self._tokenizer) 
            output = unmasker(mask_text)
            candicate_list = list(map(lambda element : element['token_str'], output))
            print("generate_mask_substitution result = {}".format(candicate_list))
            substitution_list.append(SubstitutionCandidate(token, candicate_list, [sentence_index, position_list[0], position_list[1]], mask_text))

        return substitution_list


