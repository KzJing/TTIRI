import sys

import numpy as np
import sklearn


def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray, signal: str):
    if logits.ndim == 3:
        if logits.shape[-1] == 1:
            probabilities = sigmoid(logits[..., 0])
        elif logits.shape[-1] == 2:
            probabilities = softmax(logits)[..., 1]
        else:
            print('the last ndim of logits in the metric function is an error!')
            sys.exit(1)
    else:
        print('the ndim of logits in the metric function is an error!')
        sys.exit(1)

    predictions = (probabilities >= 0.5).astype(int)

    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()
    flat_probabilities = probabilities.flatten()

    # Getting valid predicted tokens based on label information
    mask_labels = flat_labels != -100
    valid_probabilities = flat_probabilities[mask_labels]
    valid_predictions = flat_predictions[mask_labels]
    valid_labels = flat_labels[mask_labels]

    total_valid_elements = len(valid_labels)
    count_ones = np.count_nonzero(valid_labels)  # Counting the number of infidelity labels
    # print('\nClasses statistics in the current evaluation ({}-level):'.format(signal))
    # print('Total labels: {}'.format(total_valid_elements))
    # print('labels == 1: {}'.format(count_ones))
    # print('labels == 0: {}'.format(total_valid_elements - count_ones))
    # print('Distortion label ratio: {:.3f}%'.format((count_ones / total_valid_elements) * 100))

    accuracy = sklearn.metrics.accuracy_score(valid_labels, valid_predictions)
    f1 = sklearn.metrics.f1_score(valid_labels, valid_predictions)
    matthews_correlation = sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions)
    precision = sklearn.metrics.precision_score(valid_labels, valid_predictions)
    recall = sklearn.metrics.recall_score(valid_labels, valid_predictions)
    auroc = sklearn.metrics.roc_auc_score(valid_labels, valid_probabilities)
    auprc = sklearn.metrics.average_precision_score(valid_labels, valid_probabilities)

    if signal == 'sequence':
        prefix = 'seq_'
    elif signal == 'token':
        prefix = 'tok_'
    else:
        prefix = signal

    return {
        prefix + "DLR": (count_ones / total_valid_elements) * 100,
        prefix + "accuracy": accuracy,
        prefix + "f1": f1,
        prefix + "matthews_correlation": matthews_correlation,
        prefix + "precision": precision,
        prefix + "recall": recall,
        prefix + "auroc": auroc,
        prefix + "auprc": auprc
    }




# 利用评估类 来替换评估函数 以获取更多的信息传递
class CalculateMetric:

    def __init__(self, config):
        assert config.predicting_content in ['sequence', 'token', 'seq_token']
        assert config.classifier_order in ['serial', 'parallel']
        self.predicting_content = config.predicting_content
        self.classifier_order = config.classifier_order
        self.eval_seq_tok_correlation = config.eval_seq_tok_correlation


    def __call__(self, eval_pred):  # *args, **kwargs
        # Instead of passing in tensor objects, the function is passing in numpy objects

        logits, labels = eval_pred
        if self.classifier_order == 'serial':
            seq_logits, seq_labels, tok_logits, tok_labels = logits
            sequence_results = calculate_metric_with_sklearn(seq_logits, seq_labels, 'sequence')

            activation_logits = sigmoid(seq_logits).flatten()  # : (b 1 1) -> (b)
            mask_whether_pre_tok = activation_logits >= 0.5

            if np.sum(mask_whether_pre_tok, axis=0).item() > 0:
                mask_tok_logits = tok_logits[mask_whether_pre_tok]
                mask_tok_labels = tok_labels[mask_whether_pre_tok]
                token_results = calculate_metric_with_sklearn(mask_tok_logits, mask_tok_labels, 'token')

            combined_results = {}
            combined_results.update(sequence_results)
            if np.sum(mask_whether_pre_tok, axis=0) > 0:
                combined_results.update(token_results)
            return combined_results
        # eles: self.classifier_order == 'parallel':

        # logits (b s h)
        # labels (b s)

        if isinstance(logits, tuple):
            if len(logits) == 2:
                logits, labels = logits
            else:
                logits = logits[0]

        if self.predicting_content == 'sequence':
            return calculate_metric_with_sklearn(logits, labels, 'sequence')
        elif self.predicting_content == 'token':
            return calculate_metric_with_sklearn(logits, labels, 'token')
        else:  # seq_token
            sequence_logits = logits[:, :1, :]  # cls (b, 1, 1)
            sequence_labels = labels[:, :1]  # (b, 1)
            sequence_results = calculate_metric_with_sklearn(sequence_logits, sequence_labels, 'sequence')

            token_logits = logits[:, 1:, :]  # (b, s-1, 1)
            token_labels = labels[:, 1:]  # (b, s-1)

            if self.eval_seq_tok_correlation:
                activation_logits = sigmoid(sequence_logits).flatten()  # : (b 1 1) -> (b)
                mask_whether_pre_tok = activation_logits >= 0.5

                if np.sum(mask_whether_pre_tok, axis=0).item() > 0:
                    mask_tok_logits = token_logits[mask_whether_pre_tok]
                    mask_tok_labels = token_labels[mask_whether_pre_tok]
                    token_results = calculate_metric_with_sklearn(mask_tok_logits, mask_tok_labels, 'token')

                combined_results = {}
                combined_results.update(sequence_results)
                if np.sum(mask_whether_pre_tok, axis=0) > 0:
                    combined_results.update(token_results)
                return combined_results

            token_results = calculate_metric_with_sklearn(token_logits, token_labels, 'token')

            combined_results = {}
            combined_results.update(sequence_results)
            combined_results.update(token_results)
            return combined_results




