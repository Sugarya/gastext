import random
from typing import Any

# 随机游走，生成候选对抗样本
class RandomWalkTransfomer:

    def __init__(self, victim_model) -> None:
        self._victim_model = victim_model


    def __call__(self, *args: Any, **kwds: Any) -> Any:

        substitution_list = args["substitution_list"]
        real_label = args["real_label"]


        pass

    def _reverse_replace(self):

        pass



    def _random_one_round(self, substitution_list, real_label):
        substitution_size = len(substitution_list)
        random_size = 3
        count_down_size = 10
        while random_size < substitution_size:
            for count_down in range(count_down_size, 0, -1):
                array_index_list = random.sample(range(0, substitution_size), random_size)
                print("random_one_round array_index_list = {}".format(array_index_list))
 
                for index in array_index_list:
                    # 随机位置对应的同义词集
                    substitution = substitution_list[index]
                    candidate_tokens = substitution.candidate_tokens
                    position_list = substitution.postion_list
                    sentences = substitution.sentences

                    candidate_index = random.sample(range(0, len(candidate_tokens)), 1)
                    # 该同义词集里随机选取一个词
                    candidate_token = candidate_tokens[candidate_index]
                    print("random_one_round candidate_token = {}".format(candidate_token))

                    ## TODO 对原始文本做替换




                if count_down <= 0:
                    random_size *= 2
    

    def _check(self):
        pass