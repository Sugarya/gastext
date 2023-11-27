import random
from typing import Any
import numpy as np
from .estimate import LikelihoodEstimator

# 随机游走，生成候选对抗样本
class RandomWalkTransfomer:

    def __init__(self, victim_model):
        self._estimator = LikelihoodEstimator(victim_model)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # original_token，candidate_tokens，position_list，original_sentences
        substitution_list = args[0]
        origin_sentences = args[1]
        real_label = args[2]
        
        self._estimator._set_real_label(real_label)
        
        # 1. 两次随机，得到变换后的文本
        # 2. 验证变换文本是否是潜在对抗样本
        candidate_sentences, selection_list = self._random(substitution_list, real_label, origin_sentences)
        print("RandomWalkTransfomer__call__  selection_list = {}".format(selection_list))
        print("RandomWalkTransfomer__call__  candidate_sentences = {}".format(candidate_sentences))

        # 3. 反向替换，得到满足约束条件的对抗样本
        adversarial_sentences = self._reverse(candidate_sentences, selection_list)
        print("RandomWalkTransfomer__call__  adversarial_sentences = {}".format(adversarial_sentences))

        return adversarial_sentences
        
    '''
        贪心 逐个找概率最大的
        selection_list:[ ( sentence_index, candidate, origin_token) ]
    '''
    def _reverse(self, candidate_sentence_list, selection_list):
        if len(selection_list) <= 0:
            print("************************* FAILURE ********************")
            return []
        
        local_selection_list = [*selection_list]
        new_sentence_list = [*candidate_sentence_list]
        while len(local_selection_list) > 0:
            max_score = -10000
            max_index = 0
            max_sentences = []
            for i, selection in enumerate(local_selection_list):
                local_sentence_list = [*new_sentence_list]
                sentence_index, candidate, origin_token = selection[0], selection[1], selection[2]
                local_sentence_list[sentence_index].replace(candidate, origin_token)
                enable, score = self._estimator.get_likelihood_score(local_sentence_list)
                
                print("RandomWalkTransfomer _reverse enable = {}, score = {}".format(enable, score))
                if enable:
                    print("RandomWalkTransfomer _reverse score = {}".format(score))
                    if score > max_score:
                        max_score = score
                        max_index = i
                        max_sentences = [*local_sentence_list]
                        print("RandomWalkTransfomer _reverse max_score = {}".format(max_score))
            if max_score > -10000:
                del local_selection_list[max_index]
                new_sentence_list = max_sentences
                print("*************************************** REVERSE SUCCESS ********************")
                
                perturbation_count = 0
                for selection in local_selection_list:
                    perturbation_count += len(selection[2].split(' '))
            else:
                break

        return new_sentence_list
  
    '''
    组合问题，搜索 
    1. num从大到小，(n, num)组合问题，得到num个substitution

    2. 使用num个substitution变换文本，产生多个分类标签不变的候选
    3. 和变换前的输入文本的求相对熵，最大的表示为候选样本
    4. 如果候选样本词改变量数量不大于3，则为对抗样本，如果改变量数量大于3，则进入微调，继续执行random->reverse
    ''' 
    def _reverse2(self, candidate_sentence_list, substitution_list, predict_plogits):
        local_substitution_list = [*substitution_list]


        
        pass


    def _random(self, substitution_list, real_label, origin_sentences):
        substitution_size = len(substitution_list)
        random_size = 4
        count_down_size = 60 - random_size * 2
        while random_size < substitution_size:
            for count_down in range(count_down_size, 0, -1):
                sentence_list = [*origin_sentences]
                
                array_index_list = random.sample(range(0, substitution_size), random_size)
                print("RandomWalkTransfomer _random array_index_list = {}".format(array_index_list))
                
                select_sentence_index_list = []
                select_candidate_list = []
                select_origin_token_list = []
                for index in array_index_list:
                    substitution = substitution_list[index]

                    candidate_token_list, original_token = substitution.candidate_tokens, substitution.original_token
                    select_origin_token_list.append(original_token)
                    
                    candidate_index = random.sample(range(0, len(candidate_token_list)), 1)[0]
                    select_candidate = candidate_token_list[candidate_index]
                    select_candidate_list.append(select_candidate)

                    sentence_index = substitution.position_list[0]
                    print("RandomWalkTransfomer random_one_round sentence_list[sentence_index] = {}".format(sentence_list[sentence_index]))
                    sentence_list[sentence_index] = sentence_list[sentence_index].replace(original_token, select_candidate, 1)
                    print("RandomWalkTransfomer random_one_round after sentence_list[sentence_index] = {}".format(sentence_list[sentence_index]))
                    select_sentence_index_list.append(sentence_index)

                predict_probs, predict_label = self._estimator.get_predict(sentence_list)
                print("RandomWalkTransfomer random_one_round predict_label = {}, real_label = {}".format(predict_label, real_label))
                if predict_label != real_label:
                    print("**************************************************FIND IT****************************")
                    self._estimator._init_estimate(predict_probs, predict_label)
                    selection_list = list(zip(select_sentence_index_list, select_candidate_list, select_origin_token_list))
                    return sentence_list, selection_list

                if count_down <= 1:
                    random_size *= 2
                print("RandomWalkTransfomer _random count_down = {}, random_size = {}, substitution_size = {}".format(count_down, random_size, substitution_size))    
        return origin_sentences, []
    
    def __random_one_round(self, substitution, sentences):
        # 在同义词集里随机选取的词
        candidate_token_list, original_token = substitution.candidate_tokens, substitution.original_token
        candidate_index = random.sample(range(0, len(candidate_token_list)), 1)
        select_candidate_token = candidate_token_list[candidate_index]
        print("random_one_round candidate_token = {}".format(select_candidate_token))
        
        sentence_index = substitution.postion_list[0]
        sentences[sentence_index] = sentences[sentence_index].replace(original_token, select_candidate_token, 1)
