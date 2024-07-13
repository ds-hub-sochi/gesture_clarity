from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.lemmatizer import LemmatizerInterface
from src.similarity import SimilarityWrapperInterface
from src.clap_rules import ClapRulesWrapperInterface

def plot_accuracy_over_class(
    accuracy_over_label: pd.Series,
    show: bool = False,
    save_dir_path: pathlib.Path | str | None = None,
) -> None:
    columns = list(accuracy_over_label.index)
    title = 'Accuracy относительно класса \n' + \
            f'{accuracy_over_label.type} \n ' + \
            f'Среднее значение accyracy {round(accuracy_over_label.mean_accuracy, 3)}'

    columns.remove('type')
    columns.remove('mean_accuracy')

    label2value: dict[str, float] = {}
    for label in columns:
        label2value[label] = accuracy_over_label[label]
    label2value = dict(sorted(label2value.items(), key=lambda item: item[1]))

    plt.bar(list(label2value.keys()), list(label2value.values()))
    plt.xticks(rotation=90)

    plt.title(title)

    if show:
        plt.show()

    if save_dir_path is not None:
        plt.savefig(
            str(pathlib.Path(save_dir_path).joinpath(f"{accuracy_over_label['type'].replace(' ', '_')}.png")),
            bbox_inches="tight",
        )
    plt.close()


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
        gesture2homonym: dict[str, list[str]] | type[ClapRulesWrapperInterface] | None = None,
        similarity_wrapper: type[SimilarityWrapperInterface] | None = None,
    ) -> dict[str, float]:
        if gesture2homonym is None:
            gesture2homonym = {}
    
        class_accuracy: dict[str, float] = {}
        
        for file in ground_truth: # pylint: disable=[too-many-nested-blocks]
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
