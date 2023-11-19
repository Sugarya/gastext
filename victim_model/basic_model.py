import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    sep_token = ''

    def __call__(self, example):
        text = []
        if example and len(example) >= 2:
            text = example[1]
        logits = self._forward(text)
        output =  np.concatenate(logits, axis = 0)
        # print("BaseModel _forward output = {}".format(output))
        return output

    @abstractmethod
    def _forward(self, text_list):
        raise NotImplementedError
