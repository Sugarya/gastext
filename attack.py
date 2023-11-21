from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import load_data
from victim_model import HuggingFaceWrapper
from segmentation import RuleBasedExtract, SpacyProcessor, Separation
from utils import parse_arguments
from synonym import SubstitutionListCombination
from config import VICTIMS, DEVICES
from adversary_transform import RandomWalkTransfomer
from evaluation_metrics import start_evaluation, fresh_evaluation, get_calculation_list, calculate_metrics

if __name__ == '__main__':
    args = parse_arguments()
    dataset_name, victim_path = args.dataset, VICTIMS[args.victim]

    # 初始化 被攻击的模型
    classifier = AutoModelForSequenceClassification.from_pretrained(victim_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(victim_path, use_fast=True)
    victim_model_wrapper = HuggingFaceWrapper(classifier, tokenizer)

    # 初始化 短语提取
    # rule_based_extract = RuleBasedExtract()
    spacy_processor =  SpacyProcessor()
    # 划分器
    separation = Separation(victim_model_wrapper)
    # 初始化 同义词词集处理器
    generate_substitution = SubstitutionListCombination()

    # 初始化文本转换器
    # random_walk_transform = RandomWalkTransfomer(victim_model_wrapper)

    # 加载数据,生成对抗样本
    origin_examples = load_data(dataset_name)
    for index, example in enumerate(origin_examples):
        real_label = example[0]
        text = example[1]
        print(f"-------------------------------start--------------------------text = {text}")
        origin_unit_list, token_list = spacy_processor.tokenize(text)
        separation(text, token_list, unit_list = origin_unit_list)
        # origin_phrases, local_sentences, word_lists = rule_based_extract.extract_example(text)
        


        # start_evaluation(local_sentences)
        # substitution_list = generate_substitution(unit_list = origin_unit_list)
        # adversarial_sentences = random_walk_transform(substitution_list, local_sentences, real_label)
        # fresh_evaluation(adversarial_sentences)
    
    # calculate_metrics(get_calculation_list())

