import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

nlp = spacy.load('en_core_web_sm')
treebank_word_detokenizer = TreebankWordDetokenizer()

def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def detokenize(tokens):
    return treebank_word_detokenizer.detokenize(tokens)

# # substition: original_token,candidate_tokens, position_list
# def replace_origin(sentence, origin_token, candidate_token):
#     sentence.replace(origin_token, candidate_token, 1)
#     return sentence

# def replace_candidate(sentence, origin_token, candidate_token):
#     sentence.replace(candidate_token, origin_token, 1)
#     return sentence