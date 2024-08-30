from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.lemmatizer import LemmatizerInterface
from src.utils.similarity import SimilarityWrapperInterface
from src.utils.clap_rules import ClapRulesWrapperInterface


class Validator:
    def __init__(
        self,
        lemmatizer: LemmatizerInterface,
        gesture2homonyms: ClapRulesWrapperInterface,
        similarity_wrapper: SimilarityWrapperInterface,
        corrupted_cases_healer: dict[str, str] | None = None,
    ):
        self._lemmatizer: LemmatizerInterface = lemmatizer
        self._gesture2homonyms: ClapRulesWrapperInterface = gesture2homonyms
        self._similarity_wrapper: SimilarityWrapperInterface = similarity_wrapper
        self._corrupted_cases_healer: corrupted_cases_healer = corrupted_cases_healer

    def get_accuracy_over_label(  # pylint: disable=[too-many-locals]
        self,
        markup_table: pd.DataFrame,
    ) -> dict[str, float]:    
        class_accuracy: dict[str, float] = {}

        ground_truth: list[str] = list(set(markup_table.file_name))
        
        for filename in tqdm(ground_truth): # pylint: disable=[too-many-nested-blocks]
            current_file_markup: pd.DataFrame = markup_table[markup_table.file_name == filename]
            true_labels: list[str] = [filename[:-4]]

            if self._corrupted_cases_healer is not None:
                for corrupted_case in self._corrupted_cases_healer:
                    if corrupted_case in true_labels[0]:
                        true_labels.append(
                            true_labels[0].replace(
                                corrupted_case,
                                self._corrupted_cases_healer[corrupted_case],
                            )
                        )
            true_labels_normalized: list[str] = [self._lemmatizer.lemmatize_text(label) for label in true_labels]
        
            candidates: list[str] = list(current_file_markup['OUTPUT:translation'])
            candidates = [self._lemmatizer.lemmatize_text(candidate) for candidate in candidates]
    
            for candidate in candidates:
                found: bool = False
                for true_label_normalized, true_label in zip(true_labels_normalized, true_labels):
                    homonyms_set: set[str] = set(self._gesture2homonyms.get(true_label_normalized))
                    homonyms_set.update(set(self._gesture2homonyms.get(true_label)))

                    for ground_truth_homonym in list(homonyms_set):
                        for candidate_homonym in self._gesture2homonyms.get(candidate):
                            if self._similarity_wrapper.is_similar(
                                candidate_homonym,
                                ground_truth_homonym,
                            ):
                                if true_labels[0] not in class_accuracy:
                                    class_accuracy[true_labels[0]] = 0
                                class_accuracy[true_labels[0]] += 1
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
                if true_labels[0] not in class_accuracy:
                    class_accuracy[true_labels[0]] = 0

            class_accuracy[true_labels[0]] /= current_file_markup.shape[0]

        return class_accuracy

    def get_hit_table(
        self,
        markup_table: pd.DataFrame,
    ) -> np.ndarray:
        ground_truth: list[str] = list(set(markup_table.file_name))
        n_markers: int = len(list(set(markup_table.marker_id)))

        hit_table: np.ndarray = np.zeros(
            (
                len(ground_truth),
                n_markers,
            )
        )

        for i, filename in enumerate(tqdm(ground_truth)):  # pylint: disable=[too-many-nested-blocks]
            current_file_markup: pd.DataFrame = markup_table[markup_table.file_name == filename]
            true_labels: list[str] = [filename[:-4]]

            if self._corrupted_cases_healer is not None:
                for corrupted_case in self._corrupted_cases_healer:
                    if corrupted_case in true_labels[0]:
                        true_labels.append(
                            true_labels[0].replace(
                                corrupted_case,
                                self._corrupted_cases_healer[corrupted_case],
                            )
                        )
            true_labels_normalized: list[str] = [self._lemmatizer.lemmatize_text(label) for label in true_labels]
        
            candidates: list[str] = list(current_file_markup['OUTPUT:translation'])
            candidates = [self._lemmatizer.lemmatize_text(candidate) for candidate in candidates]
    
            for marker_id, candidate in zip(
                current_file_markup.marker_id,
                candidates,
            ):
                found: bool = False
                for true_label_normalized, true_label in zip(true_labels_normalized, true_labels):
                    homonyms_set: set[str] = set(self._gesture2homonyms.get(true_label_normalized))
                    homonyms_set.update(set(self._gesture2homonyms.get(true_label)))

                    for ground_truth_homonym in list(homonyms_set):
                        for candidate_homonym in self._gesture2homonyms.get(candidate):
                            if self._similarity_wrapper.is_similar(
                                candidate_homonym,
                                ground_truth_homonym,
                            ):
                                hit_table[
                                    i,
                                    int(marker_id.split('_')[-1])
                                ] = 1.0
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

        return hit_table
