from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.lemmatizer import LemmatizerInterface
from src.similarity import SimilarityWrapperInterface
from src.clap_rules import ClapRulesWrapperInterface

def plot_accuracy_over_class(
    accuracy_over_class: dict[str, float],
    title: str,
    show: bool = False,
    save: bool = True,
    save_path: str | None = None,
) -> None:
    plt.bar(list(accuracy_over_class.keys()), list(accuracy_over_class.values()))
    plt.xticks(rotation=90)

    average_accuracy: float = np.mean(list(accuracy_over_class.values()))
    plt.title(title + '\n' + f'average accuracy = {round(average_accuracy, 3)}')

    if show:
        plt.show()

    if save:
        plt.savefig(str(pathlib.Path(save_path).absolute()))


class Validator:
    def __init__(
        self,
        lemmatizer: type[LemmatizerInterface],
    ):
        self._lemmatizer: type[LemmatizerInterface] = lemmatizer

    def get_accuracy_over_label(
        self,
        ground_truth: list[str],
        markup_table: pd.DataFrame,
        gesture2homonym: dict[str, list[str]] | type[ClapRulesWrapperInterface] = dict(),
        similarity_wrapper: type[SimilarityWrapperInterface] | None = None,
    ) -> dict[str, float]:
        class_accuracy: dict[str, float] = {}
        for file in ground_truth:
            current_file_markup: pd.DataFrame = markup_table[markup_table.file_name == file]
            true_label: str = file[:-4]
        
            candidates: list[str] = list(current_file_markup['OUTPUT:translation'])
            candidates = [self._lemmatizer.lemmatize_text(candidate) for candidate in candidates]
    
            if similarity_wrapper is None:
                class_accuracy[true_label] = candidates.count(true_label)
            else:
                for candidate in candidates:
                    found = False
                    for ground_truth_homonym in gesture2homonym.get(true_label, [true_label]):
                        for candidate_homonym in gesture2homonym.get(candidate, [candidate]):
                            if similarity_wrapper.is_similar(candidate_homonym, ground_truth_homonym):
                                if true_label not in class_accuracy:
                                    class_accuracy[true_label] = 0
                                class_accuracy[true_label] += 1
                                found = True
                                break
                        if found:
                            break
            
                if true_label not in class_accuracy:
                    class_accuracy[true_label] = 0

            class_accuracy[true_label] /= current_file_markup.shape[0]

        return class_accuracy
