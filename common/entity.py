import attr

@attr.s
class OriginalPhrase:
    token = attr.ib()
    pos_tag = attr.ib()
    sentence_index = attr.ib()
    position_list = attr.ib() # [start_index, end_index]


@attr.s
class WordnetSubstitution:
    original_token = attr.ib() # ''
    candidate_tokens = attr.ib() # ['', ''], 同义词集
    position_list = attr.ib() # original_token在文本中的位置 [sentent_index, start_index, end_index]，

@attr.s
class MaskSubstitution:
    original_token = attr.ib() # ''
    candidate_tokens = attr.ib() # ['', ''], 同义词集
    position_list = attr.ib() # original_token在文本中的位置 [sentent_index, start_index, end_index]，
    mask_example = attr.ib()
    

@attr.s
class Substitution:
    original_token = attr.ib() # ''
    candidate_tokens = attr.ib() # ['', ''], 同义词集
    position_list = attr.ib() # original_token在文本中的位置 [sentent_index, start_index, end_index]，
    original_sentences = attr.ib() # [], 句子列表
    