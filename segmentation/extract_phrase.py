import re
import attr
import numpy as np
from string import punctuation as punc
import nltk
from nltk.tree import Tree
from spacy.lang.en import English
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from common import OriginPhrase

def _add_indices_to_terminals(tree):
    tree = Tree.fromstring(tree)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location]
        tree[tree_location] = non_terminal + ",/a" + str(idx)
    return str(tree)


def _extract_phrase(phrases, tree_str, label, i, stop_words_set):
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if len(subtree.label()) != 0:
                if subtree.label()[-1] == label:
                    # check the depth of the phrase
                    if subtree.height() <= 4:
                        t = subtree
                        tree_label = t.label()
                        leaves = t.leaves()
                        # if punctuation in the end of the phrase, delete it
                        if t.leaves()[-1].split(",/a")[0] in punc:
                            leaves = t.leaves()[:-1]
                            if len(leaves) == 0:
                                continue
                        start_index = leaves[0].split(",/a")[1]
                        end_index = leaves[-1].split(",/a")[1]
                        t = " ".join([leave.split(",/a")[0] for leave in leaves])
                        # check the stop_words
                        if t.strip().lower() not in stop_words_set:
                            phrases.append([t, tree_label, i, [int(start_index), int(end_index)]])

    return phrases

def _process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    return string

def _extract_phrases(parser, local_sentences, stop_words_set):
    phrases = []
    word_lists = []
    for i, sent in enumerate(local_sentences):
            tree_str = parser.parse(sent)
            tree_str = _add_indices_to_terminals(tree_str)
            word_list = [leave.split(",/a")[0] for leave in Tree.fromstring(tree_str).leaves()]
            for word_i, word in enumerate(word_list):
                if word == "-LRB-":
                    word_list[word_i] = "("
                if word == "-RRB-":
                    word_list[word_i] = ")"

            word_lists.append(word_list)
            local_sentences[i] = " ".join(word_list)
            local_sentences[i] = local_sentences[i].replace("-LRB-", "(")
            local_sentences[i] = local_sentences[i].replace("-RRB-", ")")

            phrases = _extract_phrase(phrases, tree_str, "P", i, stop_words_set)
            # replace special tokens "-LRB" and "-RRB"
            for i, phrase in enumerate(phrases):
                phrases[i][0] = phrase[0].replace("-LRB-", "(")
                phrases[i][0] = phrase[0].replace("-RRB-", ")")
    
    phrases = list(map(lambda element : OriginPhrase(element[0], element[1], element[2], element[3]), phrases))
    return phrases, local_sentences, word_lists


class RuleBasedExtract:

    def __init__(self):
        self._sentencizer = English()
        self._sentencizer.add_pipe("sentencizer")
        self._parser = StanfordCoreNLP("../stanford-corenlp-4.2.2")
        self._stop_words_set = set(nltk.corpus.stopwords.words('english'))

    def extract_example_list(self, example_list):
        all_phrases = []
        all_sentences = []
        all_word_lists = []
        for example in tqdm(example_list):
            phrases, local_sentences, word_lists = self.extract_example(example)
            all_phrases.append(phrases)
            all_sentences.append(local_sentences)
            all_word_lists.append(word_lists)

        return all_phrases, all_sentences, all_word_lists
    
    def extract_example(self, example):
        text = example[1]
        # romove the sepcial token "%" as it will lead to error of consitency parsing
        text = text.replace("%", "")
        text = text.replace(" '", "'")
        text = text.replace("$", "")
        text = _process_string(text)
        # print("extract_example text = {}".format(text))
        # split the whole example into multiple single sentences
        sents = self._sentencizer(text).sents
        local_sentences = [ sent.text for sent in sents ]
        phrases, local_sentences, word_lists = _extract_phrases(self._parser, local_sentences, self._stop_words_set)
        print("examples2phrases phrases = {}".format(phrases))
        # print("examples2phrases local_sentences = {}".format(local_sentences))
        return phrases, local_sentences, word_lists



