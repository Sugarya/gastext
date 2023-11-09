import attr

@attr.s
class SubstitutionCandidate:
    original_token = attr.ib() # ''
    candidate_token = attr.ib() # ['', '']
    position_list = attr.ib() # [sentent_index, start_index, end_index]
    mask_example = attr.ib()
    

@attr.s
class OriginalPhrase:
    token = attr.ib()
    pos_tag = attr.ib()
    sentence_index = attr.ib()
    position_list = attr.ib() # [start_index, end_index]
   