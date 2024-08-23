from __future__ import annotations

import json
import os
import pathlib
import subprocess

import click
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.validation import Validator, plot_accuracy_over_class
from src.utils.lemmatizer import NatashaBasedLemmatizer
from src.utils.similarity import NatashaSimilarityWrapper
from src.utils.clap_rules import ClapRulesWrapper

sns.set()


@click.command()
@click.argument('markup_table_path', type=click.Path(exists=True))
@click.argument('clap_rules_path', type=click.Path(exists=True))
@click.argument('clap_rules_extended_path', type=click.Path(exists=True))
@click.argument('similarity_rate', type=float)
def validate_markup(  # pylint: disable=[too-many-locals]
    markup_table_path,
    clap_rules_path,
    clap_rules_extended_path,
    similarity_rate,
) -> None:
    repository_dir_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
    print(repository_dir_path)

    embeddings_dump_path: pathlib.Path = repository_dir_path / './data/additional/embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'
    if not os.path.exists(embeddings_dump_path):
        _ = subprocess.run(
            [
                'wget',
                'https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar',
                '-O',
                f'{str(embeddings_dump_path)}',
            ],
            check=False,
        )

    similarity_wrapper: NatashaSimilarityWrapper = NatashaSimilarityWrapper(embeddings_dump_path, similarity_rate)

    lemmatizer: NatashaBasedLemmatizer = NatashaBasedLemmatizer()
    validator: Validator = Validator(lemmatizer)

    markup_table: pd.DataFrame = pd.read_csv(markup_table_path, sep='\t')
    unique_files: list[str] = list(set(markup_table.file_name))

    labels = [label[:-4] for label in unique_files]
    result_dct: dict[str, list[float] | list[str]]  = {}

    result_dct["type"] = []
    for label in labels:
        result_dct[label] = []
    result_dct["mean_accuracy"] = []
    
    with open(clap_rules_path, 'r', encoding='utf-8') as f:
        clap_rules_dct = json.load(f)
        clap_rules_wrapper: ClapRulesWrapper = ClapRulesWrapper(lemmatizer, clap_rules_dct)
    
    accuracy_over_label: dict[str, float] = validator.get_accuracy_over_label(unique_files, markup_table, clap_rules_wrapper, similarity_wrapper)
    
    result_dct["type"].append("c использованием словаря схлопываний и семантической близости")
    for label in labels:
        if label not in result_dct:
            result_dct[label] = []
        result_dct[label].append(accuracy_over_label[label])
    result_dct["mean_accuracy"].append(np.mean(list(accuracy_over_label.values())))
        
    with open(clap_rules_extended_path, 'r', encoding='utf-8') as f:
        clap_rules_dct = json.load(f)
        clap_rules_wrapper: ClapRulesWrapper = ClapRulesWrapper(lemmatizer, clap_rules_dct)

    accuracy_over_label: dict[str, float] = validator.get_accuracy_over_label(unique_files, markup_table, clap_rules_wrapper, similarity_wrapper)    
    
    result_dct["type"].append("c использованием расширенного словаря схлопываний и семантической близости")
    for label in labels:
        result_dct[label].append(accuracy_over_label[label])
    result_dct["mean_accuracy"].append(np.mean(list(accuracy_over_label.values())))

    tables_dir_path: pathlib.Path = repository_dir_path.joinpath("data/processed/tables")
    tables_dir_path.mkdir(parents=True, exist_ok=True)
    result_df: pd.DataFrame = pd.DataFrame.from_dict(result_dct)
    result_df.to_csv(tables_dir_path.joinpath("таблица_качества.csv"))

    plots_dir_path: pathlib.Path = repository_dir_path.joinpath("data/processed/plots")
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(result_df.shape[0]):
        plot_accuracy_over_class(result_df.loc[i], True, plots_dir_path)
    

if __name__ == '__main__':
    validate_markup()  # pylint: disable=[no-value-for-parameter]
