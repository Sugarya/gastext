import torch



DATASETS = {
    'test': 'test.tsv',
    'ag': 'ag.tsv',
    'qnli': 'qnli.tsv',
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