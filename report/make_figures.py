"""Publication figures for the scaffold-split EGFR experiment.

Figure 2: test ROC curves for the three learned candidates.
Figure 3: random forest confusion matrix on the scaffold test set.

Both follow the house style used for the task-similarity manuscript
(`diversity_project/src/diversity/plotting`): DejaVu Sans, semibold titles,
Wong colourblind-safe accents, the `Blues` colormap for matrices, a thin dark
panel border and a horizontal colorbar set beneath the panel. Sizes are scaled
down from that four-panel figure to a single-column figure.

Run from the repository root:

    python report/make_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

RUN = Path('reports/experiments/20261117_164744_647852_scaffold')
OUT = Path('report/figures')

# Text and furniture.
INK = '#333333'
MUTED = '#555555'
GRID = '#cfcfcf'
BOX_EDGE = '#9e9e9e'
HALO = 'white'

# Wong colourblind-safe accents. Distinguishable under deuteranopia and
# protanopia, which red/orange pairs are not.
ACCENT_BLUE = '#0077B6'
ACCENT_TEAL = '#1B9E77'
ACCENT_AMBER = '#E69F00'

HEATMAP_CMAP = 'Blues'

SERIES = {'random_forest': ACCENT_BLUE, 'knn': ACCENT_AMBER,
          'logistic': ACCENT_TEAL}
LABEL = {'random_forest': 'Random forest', 'knn': 'KNN (Tanimoto)',
         'logistic': 'Logistic regression'}

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titleweight': 'semibold',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#3f3f3f',
    'axes.labelcolor': 'black',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.alpha': 0.55,
    'legend.fontsize': 8, 'legend.edgecolor': BOX_EDGE,
    'legend.framealpha': 0.95, 'legend.fancybox': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

CLASSES = ['Non-inhibitor', 'Inhibitor']


def predictions(model):
    return pd.read_csv(RUN / f'{model}_predictions.csv')


def figure_roc():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate('Chance', xy=(0.70, 0.70), xytext=(0.74, 0.63),
                color=MUTED, fontsize=8, ha='left', va='top')

    for model, colour in SERIES.items():
        d = predictions(model)
        fpr, tpr, _ = roc_curve(d.label, d.score)
        auc = roc_auc_score(d.label, d.score)
        ax.plot(fpr, tpr, color=colour, lw=2.0, solid_capstyle='round',
                zorder=3, label=f'{LABEL[model]}  {auc:.3f}')

    ax.set_title('ROC for All Tested Models')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_box_aspect(1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(loc='lower right', title='Model and ROC-AUC',
                       borderpad=0.6, handlelength=1.6, labelspacing=0.45)
    legend.get_title().set_fontsize(8)
    legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(INK)

    save(fig, 'figure2_roc_scaffold')


def figure_confusion():
    """Counts on a `Blues` panel, matching the manuscript's matrix style.

    Rows are the true class and columns the prediction. Shading is keyed to the
    count, so darker means more molecules; the counts themselves carry the
    scale, so no colorbar is drawn. They are stroked with a white halo because
    the same label has to read on cells running from near-white to navy.
    """
    d = predictions('random_forest')
    cm = confusion_matrix(d.label, d.prediction)

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.imshow(cm, cmap=HEATMAP_CMAP, vmin=0, vmax=cm.max())

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=16, color=INK, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground=HALO),
                                  pe.Normal()])

    ax.set_title('Random Forest Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(2), CLASSES)
    ax.set_yticks(range(2), CLASSES, rotation=90, va='center')
    ax.set_box_aspect(1)

    save(fig, 'figure3_confusion_random_forest')
    return cm


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{suffix}')
    plt.close(fig)
    print(f'wrote {OUT / stem}.pdf and .png')


if __name__ == '__main__':
    figure_roc()
    cm = figure_confusion()
    tn, fp, fn, tp = cm.ravel()
    print(f'confusion: TN={tn} FP={fp} FN={fn} TP={tp}')
