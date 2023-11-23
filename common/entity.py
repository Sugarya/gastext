import attr

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
    

@attr.s
class Calculation:
    origin_sentences = attr.ib()
    adversary_sentences = attr.ib()
    query_count = attr.ib()
    perturbation_count = attr.ib()
    attack_status = attr.ib() # 1 成功，0 失败
