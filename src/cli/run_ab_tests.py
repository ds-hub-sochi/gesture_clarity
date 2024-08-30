from __future__ import annotations

import json
import os
import pathlib
import subprocess

import click
import numpy as np
import pandas as pd
import scipy.stats as sps

from src.utils.graphs import plot_accuracy_over_class
from src.utils.validation import Validator
from src.utils.lemmatizer import NatashaBasedLemmatizer
from src.utils.similarity import NatashaSimilarityWrapper
from src.utils.clap_rules import ClapRulesWrapper


@click.command()
@click.argument('control_markup_path', type=click.Path(exists=True))
@click.argument('test_markup_path', type=click.Path(exists=True))
@click.argument('clap_rules_path', type=click.Path(exists=True))
@click.argument('corrupted_cases_path', type=click.Path(exists=True))
@click.argument('similarity_rate', type=float)
@click.argument('test_alpha', type=float)
def run_ab_tests(  # pylint: disable=[too-many-locals]
    control_markup_path: str,
    test_markup_path: str,
    clap_rules_path: str,
    corrupted_cases_path: str,
    similarity_rate: float,
    test_alpha: float,
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

    similarity_wrapper: NatashaSimilarityWrapper = NatashaSimilarityWrapper(
        embeddings_dump_path,
        similarity_rate,
    )

    lemmatizer: NatashaBasedLemmatizer = NatashaBasedLemmatizer()

    with open(
        corrupted_cases_path,
        'r',
        encoding='utf-8',
    ) as f:
        corrupted_cases_healer: dict[str, str] = json.load(f)
    
    with open(
        clap_rules_path,
        'r',
        encoding='utf-8',
    ) as f:
        clap_rules_dct = json.load(f)
        clap_rules_wrapper: ClapRulesWrapper = ClapRulesWrapper(
            lemmatizer,
            clap_rules_dct,
        )

    validator: Validator = Validator(
        lemmatizer,
        clap_rules_wrapper,
        similarity_wrapper,
        corrupted_cases_healer,
    )

    control_markup_table: pd.DataFrame = pd.read_csv(
        control_markup_path,
        sep='\t',
    )
    
    test_markup_table: pd.DataFrame = pd.read_csv(
        test_markup_path,
        sep='\t',
    )

    control_hit_table: np.ndarray = validator.get_hit_table(
        control_markup_table,
    )
    control_hit_table = np.mean(control_hit_table, axis=1, keepdims=True)
    
    test_hit_table: np.ndarray = validator.get_hit_table(
         test_markup_table,
    )
    test_hit_table = np.mean(test_hit_table, axis=1, keepdims=True)

    print('T-test is runnig; checking the equal means null hypotesis: ', end='')
    ttest_results: sps._result_classes.TtestResult = sps.ttest_ind(
        control_hit_table,
        test_hit_table,
        equal_var = False,
        axis=0,
        alternative='greater',
    )
    print('H0 was rejected' if ttest_results.pvalue > test_alpha else "H0 wasn't rejected")

    print('Mann-Whitneyu test is runnig; checking the equal distrubitions null hypotesis: ', end='')
    mannwhitneyu_results: sps._result_classes.MannwhitneyuResult = sps.mannwhitneyu(
        control_hit_table,
        test_hit_table,
        axis=0,
        alternative='greater',
    )
    print('H0 was rejected' if mannwhitneyu_results.pvalue > test_alpha else "H0 wasn't rejected")


if __name__ == '__main__':
    run_ab_tests()
