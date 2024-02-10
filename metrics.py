import numpy as np
import torch
from torchmetrics import AveragePrecision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            RocCurveDisplay, auc, roc_curve, precision_recall_curve, \
                            PrecisionRecallDisplay, average_precision_score


def accuracy(y_pred_avpu, y_pred_crt, y_true_avpu, y_true_crt):
    """ Calculate the accuracy. """

    y_pred = torch.concat((torch.argmax(y_pred_avpu, axis=1), torch.argmax(y_pred_crt, axis=1)))
    y_true = torch.concat((torch.argmax(y_true_avpu, axis=1), torch.argmax(y_true_crt, axis=1)))
    return accuracy_score(y_true, y_pred)


def calculate_metrics(y_pred, y_true, average='macro'):
    """ Calculate the accuracy, precision, recall and F1 score. """

    y_pred = torch.argmax(y_pred, axis=1)
    y_true = torch.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0.0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0.0)

    return [accuracy, precision, recall, f1]


def calculate_auroc(y_pred, y_true, name, average='macro', all_classes=True, visualize=False):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC)
    using a One vs the Rest method. Average is 'macro', means each class is treated
    equally; average is 'micro' means the largest is considered more important.

    All classes = True means the AUROC is calculated across all classes, even if
    that class does not appear in the true labels.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    """

    n_classes = y_pred.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Micro-averaged
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Interpolate ROC curves at these points
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    valid_class = list(range(n_classes))
    for i in range(n_classes):
        if 1 not in y_true[:, i]: # Put missing classes at chance level
            if all_classes:
                fpr[i], tpr[i] = np.array([0., 1.]), np.array([0., 1.])
                roc_auc[i] = 0.5
            else:
                fpr[i], tpr[i] = np.array([0., 1.]), np.array([0., 0.])
                roc_auc[i] = 0
            valid_class.remove(i)

        else:
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    if all_classes:
        mean_tpr /= n_classes
    else:
        mean_tpr /= len(valid_class)

    fpr['macro'] = fpr_grid
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr["macro"], tpr["macro"])

    # Plot values
    if visualize:
        fig, ax = plt.subplots(figsize=(9, 5))

        plt.plot(
            fpr[average],
            tpr[average],
            label=f"{average}-average ROC curve (AUC = {roc_auc[average]:.2f})",
            color="red",
            linestyle="-",
            linewidth=2
        )

        for i in range(n_classes):
            RocCurveDisplay.from_predictions(
                y_true[:, i],
                y_pred[:, i],
                name=f"ROC curve for {name}={i}",
                ax=ax,
                linestyle='--',
                plot_chance_level=(i == n_classes-1),
                chance_level_kw={'alpha': 0.5},
                alpha=0.3
            )

        actual_auroc = round(np.mean([v for k, v in roc_auc.items() if k in valid_class]), 2)

        title = "ROC curve for One-vs-Rest multiclass"
        if average == 'macro' and all_classes == True:
            title += f"\n(AUROC without missing classes: {actual_auroc})"

        _ = ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=title
        )
        plt.legend(bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)
        plt.tight_layout()
        plt.show()

    return roc_auc[average]


def calculate_auprc(y_pred, y_true, name, average='macro', all_classes=True, visualize=False):
    """
    Calculate the Area Under the Precision Recall Curve (AUPRC)
    using a One vs the Rest method. Average is 'macro', means each class is treated
    equally; average is 'micro' means the largest is considered more important.

    All classes = True means the AUROC is calculated across all classes, even if
    that class does not appear in the true labels.
    """

    n_classes = y_pred.shape[1]

    precision, recall, ap = dict(), dict(), dict()

    precision['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    ap['micro'] = average_precision_score(y_true, y_pred, average='micro')

    # Interpolate PR curves at these points
    recall_grid = np.linspace(0.0, 1.0, 1000)
    mean_precision = np.zeros_like(recall_grid)

    valid_class = list(range(n_classes))
    for i in range(n_classes):
        if 1 not in y_true[:, i]: # Put missing classes at chance level
            if all_classes:
                precision[i], recall[i] = np.array([0, 0]), np.array([0., 1.])
                ap[i] = 1/n_classes
            else:
                precision[i], recall[i] = np.array([0, 0]), np.array([0., 1.])
                ap[i] = 0
            valid_class.remove(i)
        
        else:
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            ap[i] = average_precision_score(y_true[:, i], y_pred[:, i])

        mean_precision += np.interp(recall_grid, recall[i][::-1], precision[i][::-1])

    if all_classes:
        mean_precision /= n_classes
    else:
        mean_precision /= len(valid_class)
    
    recall['macro'] = recall_grid
    precision['macro'] = mean_precision
    if all_classes:
        ap['macro'] = average_precision_score(y_true, y_pred, average='macro')
    else:
        ap['macro'] = average_precision_score(y_true[:, valid_class], y_pred[:, valid_class], average='macro')

    if visualize:
        fig, ax = plt.subplots(figsize=(9, 5))

        plt.plot(
            recall[average],
            precision[average],
            label=f"{average}-average PR curve (AUC = {ap[average]:.2f})",
            color='red',
            linestyle='-',
            linewidth=2
        )

        for i in range(n_classes):
            PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=ap[i],
                estimator_name=f"PR curve for {name}={i}",
            ).plot(ax=ax, alpha=0.3, linestyle='--',)

        plt.axhline(1/n_classes, linestyle='--', color='black',
                    label=f'Chance level (AP = {1/n_classes})', alpha=0.5)

        actual_auprc = round(np.mean([v for k, v in ap.items() if k in valid_class]), 2)

        title = "ROC curve for One-vs-Rest multiclass"
        if average == 'macro' and all_classes == True:
            title += f"\n(AUROC without missing classes: {actual_auprc})"

        _ = ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=title,
        )

        plt.legend(bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)
        plt.tight_layout()
        plt.show()

    return ap[average]


