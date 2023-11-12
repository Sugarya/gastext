import torch

from ..basic_model import BaseModel


class HuggingFaceWrapper(BaseModel):
    def __init__(self, model, tokenizer):
        super(HuggingFaceWrapper, self).__init__()

        self._model = model
        self._tokenizer = tokenizer

        self.unk_token = self._tokenizer.unk_token
        self.sep_token = self._tokenizer.sep_token

    def _forward(self, text_list):
        inputs = self._tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        device = next(self._model.parameters()).device
        inputs.to(device)
        print("_forward inputs={}".format(inputs))
        with torch.no_grad():            
            outputs = self._model(**inputs)

        return outputs.logits.cpu().numpy()
