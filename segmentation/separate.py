

'''
    划分器,并计算脆弱值
'''
from typing import Any


class Separation:

    def __init__(self, victim_model) -> None:
        self._victim_model = victim_model

    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        origin_text = args[0]
        token_list = args[1]
        origin_unit_list = kwds["unit_list"]

        logits = self._victim_model(origin_text)
        print(f"Separation __call__  logits = {logits}")
        for unit in origin_unit_list:
            position = unit.origin_position
            print(f"unit.word : {unit.word} =?= {token_list[position]}")
            token_list[position] = ''
            # 如何detoken


            

        


