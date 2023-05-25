from typing import Sequence, Callable, Dict

import numpy as np
import scipy
import torch
from seqeval.metrics import accuracy_score
from transformers import EvalPrediction


def accuracy_score_remote(y_true, y_pred):
    pred_idx = np.argmax(y_pred, axis=1)
    # for y_t, y_p in zip(y_true, pred_idx):
    #     print(y_t, y_p)
    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, pred_idx))

    nb_true = len(y_true)
    score_top1 = nb_correct / nb_true

    return score_top1


def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


def compute_accuracy_metrics(task_name, preds, labels):
    if task_name == 'remote_homology':
        return {
            "accuracy": accuracy_score_remote(labels, preds)
        }
    else:
        raise KeyError(task_name)


def compute_spearmanr_metrics(task_name, preds, labels):
    # print(p.label_ids.shape, p.predictions.shape)
    if task_name == 'fluorescence' or task_name == 'stability':
        return{
            "spearmanr": spearmanr(labels, preds)
        }
    else:
        raise KeyError(task_name)


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean()


def bt_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

    # TODO: complement remain tasks' metrics
    if task_name == 'ss3' or task_name == 'ss8':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def build_compute_metrics_fn(task_name: str, output_type: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_type == 'token-level-classification':
            logits = p.predictions
            preds = np.argmax(logits, axis=-1)
            label_ids = torch.from_numpy(p.label_ids)
            preds = torch.from_numpy(preds)

            active_index = (label_ids.view(-1) != -100)
            active_preds = preds.view(-1)[active_index]
            active_labels = label_ids.view(-1)[active_index]
            return compute_metrics_mapping[task_name](task_name, active_preds, active_labels)
        elif output_type == 'sequence-level-classification' or output_type == 'sequence-level-regression':
            logits = p.predictions
            # preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids
            return compute_metrics_mapping[task_name](task_name, logits, label_ids)
        else:
            raise Exception("output type not supported.")

    return compute_metrics_fn


compute_metrics_mapping = {
    'ss3': bt_compute_metrics,
    'ss8': bt_compute_metrics,
    'remote_homology': compute_accuracy_metrics,
    'fluorescence': compute_spearmanr_metrics,
    'stability': compute_spearmanr_metrics,
    'contact': None
}
