import torch



DATASETS = {
    'test': 'test.tsv',
    'ag': 'ag.tsv',
    'qnli': 'qnli.tsv',
}

FILL_MASK_MODEL = {
    'bert-base': ('../preTrainedModel/bert-base-uncased', '../preTrainedModel/bert-base-uncased/config.json')
}

MODEL_POOL = {
    'use_path': 'preTrainedModel/universal-sentence-encoder_4/',
    'use': 'https://hub.tensorflow.google.cn/google/universal-sentence-encoder/4'
}


VICTIMS = {
    'ag': '../preTrainedModel/bert-base-uncased-ag-news/',
    'imdb': '../preTrainedModel/bert-base-uncased-imdb/',
    'yelp-polarity': '../preTrainedModel/bert-base-uncased-yelp-polarity/',
}


DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
if len(DEVICES) == 0:
    DEVICES = ['cpu'] * 2
elif len(DEVICES) == 1:
    DEVICES = DEVICES * 2



