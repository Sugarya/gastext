import json
from common import ATTACK_STATUS

class Persistence:

    def __init__(self, save_file_path) -> None:
        self._save_file_path = save_file_path
        self._count = 0


    def append(self, calculation): # CalculationEntity
        if self._save_file_path:
            if calculation._attack_status == ATTACK_STATUS.SUCCESS:
                print_dict = calculation.to_dict(self._count)
                output_path = self._save_file_path + '.json'
                with open(output_path, 'a+') as f:
                    json.dump(print_dict, f, indent = 4, ensure_ascii = False)
                    self._count = self._count + 1

