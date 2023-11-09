
from utils.argpaser import load_arguments
from dataset import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from victim_model import HuggingFaceWrapper
from config_attack import VICTIMS, DEVICES
from synonym import RuleBasedExtract

if __name__ == '__main__':
    args = load_arguments()
    dataset_name, victim_path = args.dataset, VICTIMS[args.victim]
    # print(victim_path)
    classifier = AutoModelForSequenceClassification.from_pretrained(victim_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(victim_path, use_fast=True)
    victim_model_wrapper = HuggingFaceWrapper(classifier, tokenizer)
    
    origin_examples = load_data(dataset_name)
    rule_based_extract = RuleBasedExtract()
    # all_phrases, all_sentences, all_word_lists = rule_based_extract.extract_example_list(origin_examples)

    for index, example in enumerate(origin_examples) :
        print("---------------------------------------start")
        phrases, local_sentences, word_lists = rule_based_extract.extract_example(example)
        plogits = victim_model_wrapper(local_sentences)
        print("__main__ example output = {}".format(plogits))




