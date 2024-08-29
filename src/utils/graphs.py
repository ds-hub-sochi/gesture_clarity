from __future__ import annotations

import math
import pathlib

import numpy as np
from matplotlib import pyplot as plt


def plot_accuracy_over_class(
    accuracy_over_label: dict[str, float],
    show: bool,
    save_dir_path: pathlib.Path | str | None,
    title: str,
) -> None:
    mean_accuracy = float(np.mean(list(accuracy_over_label.values())))

    label2value = dict(
        sorted(
            accuracy_over_label.items(),
            key=lambda item: item[1],
        )
    )

    plt.bar(
        list(label2value.keys()),
        list(label2value.values()),
    )

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

def plot_markups_comparison(
    control_accuracy_over_label: dict[str, float],
    test_accuracy_over_label: dict[str, float],
    show: bool,
    save_dir_path: pathlib.Path | str | None,
    title: str,
):
    common_keys: set[str] = set(control_accuracy_over_label.keys()).intersection(test_accuracy_over_label.keys())

    control_accuracy_over_label = dict(sorted(control_accuracy_over_label.items(), key=lambda item: item[1]))

    control_accuracy_filtered: dict[str, float] = {key: control_accuracy_over_label[key] for key in control_accuracy_over_label if key in common_keys}
    test_accuracy_filtered: dict[str, float] = {key: test_accuracy_over_label[key] for key in control_accuracy_filtered}

    figure_width: int = math.ceil(len(common_keys) * 0.5)
    figure_height: int = math.ceil(figure_width * 3 / 4)
    figure = plt.figure(
        figsize=(
            figure_width,
            figure_height,
        ),
        dpi=80,
    )
      
    x_ticks = np.arange(len(common_keys)) 
      
    plt.bar(
        x_ticks - 0.2,
        list(control_accuracy_filtered.values()),
        0.4,
        label = 'Контрольная выборка',
    ) 
    plt.bar(
        x_ticks + 0.2,
        list(test_accuracy_filtered.values()),
        0.4,
        label = 'Тестовая выборка',
    )

    plt.xticks(
        x_ticks,
        list(control_accuracy_filtered.keys()),
        rotation=90,
        fontsize=20,
    )
    
    plt.yticks(
        np.arange(0, 1.0 + 0.1, 0.1),
        fontsize=20,    
    )

    mean_gain: float = float(np.mean(np.array(list(test_accuracy_filtered.values())) - np.array(list(control_accuracy_filtered.values()))))

    plt.title(
        f'{title}\nСредний прирост: {round(mean_gain, 3)}',
        fontsize=20,
    ) 
    
    plt.legend(fontsize=20)

    if save_dir_path is not None:
        plt.savefig(
            str(pathlib.Path(save_dir_path).joinpath(f"график_{title.replace(' ', '_')}.png")),
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close()
    