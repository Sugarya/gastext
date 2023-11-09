import attr

@attr.s
class SubstitutionCandidate:
    original_token = attr.ib() # ''
    candidate_token = attr.ib() # ['', '']

@attr.s
class OriginPhrase:
    token = attr.ib()
    pos_tag = attr.ib()
    sentence_index = attr.ib()
    position_list = attr.ib() # [start_index, end_index]
   