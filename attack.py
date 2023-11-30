import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import load_data
from victim_model import HuggingFaceWrapper, estimate
from segmentation import RuleBasedExtract, Separation
from utils import parse_arguments, spacy_process
from synonym import SubstitutionListCombination
from config import VICTIMS, DEVICES
from adversary_transform import DynamicPlanning
from metrics import calculate, Persistence

if __name__ == '__main__':
    args = parse_arguments()
    dataset_name, victim_path, output_file_path= args.dataset, VICTIMS[args.victim], args.output
    # 初始化 被攻击的模型
    classifier = AutoModelForSequenceClassification.from_pretrained(victim_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(victim_path, use_fast=True)
    victim_model_wrapper = HuggingFaceWrapper(classifier, tokenizer)

    # 初始化 短语提取
    # rule_based_extract = RuleBasedExtract()
    # 划分器
    separation = Separation(victim_model_wrapper)
    # 初始化 同义词词集处理器
    generate_substitution = SubstitutionListCombination()

    dynamic_planning = DynamicPlanning(victim_model_wrapper)

    persistence = Persistence(output_file_path)

    # 初始化文本转换器
    # random_walk_transform = RandomWalkTransfomer(victim_model_wrapper)

    # 加载数据,生成对抗样本
    origin_examples = load_data(dataset_name)
    for index, example in enumerate(origin_examples):
        real_label, text = example[0], example[1]
        calculate.start_evaluation(text, real_label)
        origin_probs = estimate._get_probability(victim_model_wrapper, text)
        prob_label = np.argmax(origin_probs)
        if not prob_label == real_label:
            continue
        
        print(f"------------------------------- start {index} --------------------------text = {text}")
        # origin_phrases, local_sentences, word_lists = rule_based_extract.extract_example(text)
        # 1 拆解句子
        origin_unit_list = spacy_process.split(text)
        # Optional todo 粗筛后再生成同义词
        # 2 生成同义词集
        candidate_lists = generate_substitution(text, unit_list = origin_unit_list)
        origin_unit_list = separation(candidate_lists, origin_unit_list, text, real_label)
        


        # 3 计算最终的脆弱值，然后进行排序;4 查找
        adversarial_example = dynamic_planning(text, candidate_lists, origin_unit_list)
        # 统计指标
        calculate_entity = calculate.fresh_evaluation(adversarial_example)
        # persistence.append(calculate_entity)
    
    # calculate_metrics(get_calculation_list())

