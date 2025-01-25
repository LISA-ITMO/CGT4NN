from matplotlib.axes import Axes


from dataclasses import dataclass


@dataclass
class EvaluationSubplots:
    accuracy_ax: Axes
    f1_ax: Axes
    roc_auc_ax: Axes
    mse_ax: Axes
    r2_ax: Axes