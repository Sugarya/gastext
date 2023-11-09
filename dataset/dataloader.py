
import csv
from config_attack import DATASETS


def load_data(dataset_name):
    file_path = "./dataset/srcset/%s" % DATASETS[dataset_name]
    print('file_path = {}'.format(file_path))
    examples = read_corpus(file_path)

    return examples


def read_corpus(path):
    with open(path, encoding='utf8') as f:
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if len(examples[i]) == 2:
                examples[i].append(None)
    return examples
