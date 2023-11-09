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