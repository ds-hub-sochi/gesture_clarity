from __future__ import annotations

import json
import os
import pathlib
import subprocess

import click
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
@click.argument('corrupted_cases_path', type=click.Path(exists=True))
@click.argument('similarity_rate', type=float)
@click.argument('title', type=str)
def validate_markup(  # pylint: disable=[too-many-locals]
    markup_table_path: str,
    clap_rules_path: str,
    corrupted_cases_path: str,
    similarity_rate: float,
    title: str,
) -> None:
    repository_dir_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()

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

    with open(corrupted_cases_path, 'r', encoding='utf-8') as f:
        corrupted_cases_healer: dict[str, str] = json.load(f)
    
    with open(clap_rules_path, 'r', encoding='utf-8') as f:
        clap_rules_dct = json.load(f)
        clap_rules_wrapper: ClapRulesWrapper = ClapRulesWrapper(lemmatizer, clap_rules_dct)
    
    accuracy_over_label: dict[str, float] = validator.get_accuracy_over_label(
        unique_files,
        markup_table,
        clap_rules_wrapper,
        similarity_wrapper,
        corrupted_cases_healer,
    )

    table_dir_path: pathlib.Path = repository_dir_path.joinpath("data/processed/tables")
    table_dir_path.mkdir(parents=True, exist_ok=True)

    result_df: pd.DataFrame = pd.DataFrame.from_dict(
        {key: [value] for key, value in accuracy_over_label.items()}
    )
    result_df.to_csv(table_dir_path.joinpath(f"таблица_{title}.csv"))

    plot_dir_path: pathlib.Path = repository_dir_path.joinpath("data/processed/plots")
    plot_dir_path.mkdir(parents=True, exist_ok=True)
    
    plot_accuracy_over_class(accuracy_over_label, True, plot_dir_path, title)
    
if __name__ == '__main__':
    validate_markup()  # pylint: disable=[no-value-for-parameter]
