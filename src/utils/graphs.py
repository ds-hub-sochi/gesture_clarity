from __future__ import annotations

import pathlib

import numpy as np
from matplotlib import pyplot as plt


def plot_accuracy_over_class(
    accuracy_over_label: dict[str, float],
    show: bool,
    save_dir_path: pathlib.Path | str,
    title: str,
) -> None:
    mean_accuracy = float(np.mean(list(accuracy_over_label.values())))

    label2value = dict(sorted(accuracy_over_label.items(), key=lambda item: item[1]))

    plt.bar(list(label2value.keys()), list(label2value.values()))
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 1.0 + 0.1, 0.1))

    title_parts: list[str] = title.split(' ')
    if len(title_parts) > 6:
        plt.title(
            f'{" ".join(title_parts[:4])}\n' + \
            f'{" ".join(title_parts[4:])}\n' + \
            f'Среднее значение accuracy = {round(mean_accuracy, 3)}'
        )

    if show:
        plt.show()

    if save_dir_path is not None:
        plt.savefig(
            str(pathlib.Path(save_dir_path).joinpath(f"график_{title.replace(' ', '_')}.png")),
            bbox_inches="tight",
        )
    plt.close()