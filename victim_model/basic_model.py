import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    sep_token = ''

    def __call__(self, workload, batch_size=64):
        text = workload[1]
        logits = []
        # bz = batch_size
        #     batch = []
        #     for i in range((len(text) + bz - 1) // bz):
        #         batch.append(text[i * bz : (i + 1) * bz])
        #     print("_forward batch = {}".format(batch))    
        logits.append(self._forward(text))
        print("_forward logits = {}".format(logits))
        output =  np.concatenate(logits, axis=0)
        return output

    @abstractmethod
    def _forward(self, text_list):
        raise NotImplementedError
