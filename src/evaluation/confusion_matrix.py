from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    cm,
    class_names,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(cm)

    ax.set_xticks(
        np.arange(len(class_names))
    )
    ax.set_yticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "LSTM Test Confusion Matrix"
    )

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)