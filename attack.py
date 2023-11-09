from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import load_data
from victim_model import HuggingFaceWrapper
from segmentation import RuleBasedExtract
from utils import parse_arguments
from synonym import generate_wordnet_substitution, FillMaskCandidate
from config_attack import VICTIMS, DEVICES

if __name__ == '__main__':
    args = parse_arguments()
    dataset_name, victim_path = args.dataset, VICTIMS[args.victim]
    # print(victim_path)
    classifier = AutoModelForSequenceClassification.from_pretrained(victim_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(victim_path, use_fast=True)
    victim_model_wrapper = HuggingFaceWrapper(classifier, tokenizer)
    
    origin_examples = load_data(dataset_name)
    rule_based_extract = RuleBasedExtract()
    fillMaskCandidate = FillMaskCandidate()
   

    for index, example in enumerate(origin_examples) :
        print("---------------------------------------start")
        phrases, local_sentences, _ = rule_based_extract.extract_example(example)
        # print("__main__ phrases = {}".format(phrases))
        # print("__main__ local_sentences = {}".format(local_sentences))

        wordnet_substitution_list = generate_wordnet_substitution(phrases)
        # print("__main__ substitution_list = {}".format(substitution_list))
        mask_substitution_list = fillMaskCandidate.generate_mask_substitution(phrases, local_sentences)
        
        # plogits = victim_model_wrapper(local_sentences)
        # print("__main__ example output = {}".format(plogits))




