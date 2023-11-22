import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    sep_token = ''

    def __call__(self, text_list):
        text = []
        if isinstance(text_list, str):
            text.append(text_list)
        elif isinstance(text_list, list):
            text = [*text_list]
        else:
            text = text_list    
        # print(f"BaseModel _forward text = {text}")
        logits = self._forward(text)
        output =  np.concatenate(logits, axis = 0)
        
        return output

    @abstractmethod
    def _forward(self, text_list):
        raise NotImplementedError
