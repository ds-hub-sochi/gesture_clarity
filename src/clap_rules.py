from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from src.lemmatizer import LemmatizerInterface


class ClapRulesWrapperInterface(ABC):
    @abstractmethod
    def get(
        self,
        token: str,
    ) -> list[str]:
        pass

class ClapRulesWrapper(ClapRulesWrapperInterface):
    def __init__(
        self,
        lemmatizer: LemmatizerInterface,
        clap_rules_dct: dict[str, list[str]],
    ):
        super().__init__()

        self._gesture2homonym: dict[str, list[str]] = {}

        for keys in clap_rules_dct:
            for key in keys.split('/'):
                key = lemmatizer.lemmatize_text(key)
                for value in clap_rules_dct[keys]:
                    value = lemmatizer.lemmatize_text(value)
                    if key not in self._gesture2homonym:
                        self._gesture2homonym[key] = []
                    self._gesture2homonym[key].append(value)
    
                    if value not in self._gesture2homonym:
                        self._gesture2homonym[value] = []
                    self._gesture2homonym[value].append(key)

        has_difference: bool = True

        # maybe it can converge in 1 iteration only, but I'm not sure.
        while has_difference:
            gesture2homonym_copy: dict[str, list[str]] = deepcopy(self._gesture2homonym)
            
            for key, homonym_lst in self._gesture2homonym.items():
                for homonym in homonym_lst:
                    temp_lst = self._gesture2homonym[homonym] + homonym_lst
                    self._gesture2homonym[homonym] = list(set(temp_lst))
                    self._gesture2homonym[key] = list(set(temp_lst))
        
            has_difference = False
            for key, homonym_lst in self._gesture2homonym.items():
                if (sorted(homonym_lst) != sorted(gesture2homonym_copy[key])):
                    has_difference = True
                    break

    def get(
        self,
        token: str,
    ) -> list[str]:
        return self._gesture2homonym[token] if token in self._gesture2homonym else [token]
        