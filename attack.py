from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import load_data
from victim_model import HuggingFaceWrapper
from segmentation import RuleBasedExtract
from utils import parse_arguments
from synonym import SubstitutionListCombination
from config_attack import VICTIMS, DEVICES

if __name__ == '__main__':
    args = parse_arguments()
    dataset_name, victim_path = args.dataset, VICTIMS[args.victim]

    classifier = AutoModelForSequenceClassification.from_pretrained(victim_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(victim_path, use_fast=True)
    victim_model_wrapper = HuggingFaceWrapper(classifier, tokenizer)
    
    origin_examples = load_data(dataset_name)
    rule_based_extract = RuleBasedExtract()
    substitution_generate = SubstitutionListCombination()

    for index, example in enumerate(origin_examples) :
        print("---------------------------------------start")
        phrases, local_sentences, _ = rule_based_extract.extract_example(example)
        # print("__main__ phrases = {}".format(phrases))
        # print("__main__ local_sentences = {}".format(local_sentences))
        substitution_list = substitution_generate(phrases, local_sentences)

        real_label = example[0]
        


        # plogits = victim_model_wrapper(local_sentences)
        # print("__main__ example output = {}".format(plogits))




