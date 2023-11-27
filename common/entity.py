import attr
from typing import Any

class ATTACK_STATUS:
    FAILURE = 'failure'
    SUCCESS = 'SUCCESS'

@attr.s
class OriginalPhrase:
    token = attr.ib()
    pos_tag = attr.ib()
    sentence_index = attr.ib()
    position_list = attr.ib() # [start_index, end_index]

@attr.s
class OriginalUnit:
    word = attr.ib()
    lemma = attr.ib()
    pos_tag = attr.ib()
    sentence_index = attr.ib()
    origin_position = attr.ib()
    spacy_token = attr.ib()
    fragile_value = attr.ib()

@attr.s
class MaskSubstitution:
    original_token = attr.ib() # ''
    candidate_tokens = attr.ib() # ['', ''], 同义词集
    position_list = attr.ib() # original_token在文本中的位置 [sentent_index, start_index, end_index]，
    mask_example = attr.ib()

@attr.s
class NetSubstitution:
    original_token = attr.ib() # ''
    candidate_tokens = attr.ib() # ['', ''], 同义词集
    origin_position = attr.ib()

@attr.s
class Candidate:
    synonym = attr.ib() # 同义词
    fragile_value = attr.ib() # 脆弱值值
    


class CalculationEntity:
    _origin_text = ''
    _origin_label = None
    _adversary_text = ''
    _attack_label = None

    _query_count = 0
    _perturbation_count = 0
    _attack_status = ''
    _origin_tokens = []
    _substitution_tokens = []


    def __init__(self, *args: Any, **kwds: Any) -> None:
        self._origin_text = ''
        self._origin_label = None
        self._adversary_text = ''
        self._attack_label = None

        self._query_count = 0
        self._perturbation_count = 0
        self._attack_status = ''
        self._origin_tokens = []
        self._substitution_tokens = []

    def reset(self):
        self._origin_text = ''
        self._origin_label = None
        self._adversary_text = ''
        self._attack_label = None

        self._query_count = 0
        self._perturbation_count = 0
        self._attack_status = ''
        self._origin_tokens = []
        self._substitution_tokens = []


    def __str__(self) -> str:

        return f'{self._origin_text}{self._origin_label}\n{self._adversary_text}\n{self._attack_label}\n{self._query_count}\n{self._perturbation_count}\n{self._attack_status}\n{self._origin_tokens}\n{self._substitution_tokens}'
    
    def to_dict(self, index = None) -> dict:
        d = {
            'origin_text': self._origin_text,
            'origin_label': str(self._origin_label),
            'adversary_text': self._adversary_text,
            'attack_label': str(self._attack_label),
            'query_count': self._query_count,
            'perturbation_count': self._perturbation_count,
            'attack_status': self._attack_status,
            'origin_tokens': self._origin_tokens,
            'substitution_tokens': self._substitution_tokens
        }
        if index is not None:
            d['id'] = index
        return d