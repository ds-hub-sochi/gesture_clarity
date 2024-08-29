from __future__ import annotations

import pathlib

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
    ):
        self._lemmatizer: LemmatizerInterface = lemmatizer

    def get_accuracy_over_label(  # pylint: disable=[too-many-locals]
        self,
        markup_table: pd.DataFrame,
        gesture2homonyms: ClapRulesWrapperInterface,
        similarity_wrapper: SimilarityWrapperInterface,
        corrupted_cases_healer: dict[str, str] | None = None,
    ) -> dict[str, float]:    
        class_accuracy: dict[str, float] = {}

        ground_truth: list[str] = list(set(markup_table.file_name))
        
        for file in tqdm(ground_truth): # pylint: disable=[too-many-nested-blocks]
            current_file_markup: pd.DataFrame = markup_table[markup_table.file_name == file]
            true_labels: list[str] = [file[:-4]]

            if corrupted_cases_healer is not None:
                for corrupted_case in corrupted_cases_healer:
                    if corrupted_case in true_labels[0]:
                        true_labels.append(true_labels[0].replace(corrupted_case, corrupted_cases_healer[corrupted_case]))
            true_labels_normalized: list[str] = [self._lemmatizer.lemmatize_text(label) for label in true_labels]
        
            candidates: list[str] = list(current_file_markup['OUTPUT:translation'])
            candidates = [self._lemmatizer.lemmatize_text(candidate) for candidate in candidates]
    
            for candidate in candidates:
                found: bool = False
                for true_label_normalized, true_label in zip(true_labels_normalized, true_labels):
                    homonyms_set: set[str] = set(gesture2homonyms.get(true_label_normalized))
                    homonyms_set.update(set(gesture2homonyms.get(true_label)))

                    for ground_truth_homonym in list(homonyms_set):
                        for candidate_homonym in gesture2homonyms.get(candidate):
                            if similarity_wrapper.is_similar(candidate_homonym, ground_truth_homonym):
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
