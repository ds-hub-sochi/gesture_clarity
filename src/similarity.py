from __future__ import annotations

import pathlib
from abc import abstractmethod, ABC

import numpy as np
from navec import Navec


class SimilarityWrapperInterface(ABC):
    @abstractmethod
    def get_similarity(
        self,
        first_token: str,
        second_token: str,
    ) -> float:
        pass

    @abstractmethod
    def is_similar(
        self,
        first_token: str,
        second_token: str,
    ) -> bool:
        pass

class NatashaSimilarityWrapper(SimilarityWrapperInterface):
    def __init__(
        self,
        path_to_embeddings_dump: str | pathlib.Path,
        similarity_score: float,
    ):
        super().__init__()

        self._embeddings: Navec = Navec.load(path_to_embeddings_dump)
        self._similatity_score: float = similarity_score

    def get_similarity(
        self,
        first_token: str,
        second_token: str,
    ) -> float:
        if first_token in self._embeddings:
            first_emb: np.ndarray = self._embeddings[first_token]
        else:
            first_emb = self._embeddings['<unk>']

        if second_token in self._embeddings:
            second_emb: np.ndarray = self._embeddings[second_token]
        else:
            second_emb = self._embeddings['<unk>']

        return first_emb.T @ second_emb / (np.linalg.norm(first_emb) * np.linalg.norm(second_emb))

    def is_similar(
        self,
        first_token: str,
        second_token: str,
    ) -> bool:
        return self.get_similarity(first_token, second_token) > self._similatity_score
