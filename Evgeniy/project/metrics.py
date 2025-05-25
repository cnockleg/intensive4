import numpy as np
import torch

from sklearn.metrics import roc_auc_score, f1_score


def calculate_auc(labels, preds):
    auc_per_tag = [roc_auc_score(labels[:, i], preds[:, i]) for i in range(labels.shape[1])]
    micro_auc = roc_auc_score(labels.ravel(), preds.ravel())
    return auc_per_tag, micro_auc


def optimize_f1_thresholds(labels, preds, thresholds=np.linspace(0.0, 1.0, 101)):
    optimal_thresholds = []
    f1_per_tag = []
    for i in range(labels.shape[1]):
        best_f1, best_thresh = max(
            ((f1_score(labels[:, i], (preds[:, i] >= t).astype(int), zero_division=0), t) for t in thresholds),
            key=lambda x: x[0]
        )
        optimal_thresholds.append(best_thresh)
        f1_per_tag.append(best_f1)
    return optimal_thresholds, f1_per_tag


def apply_thresholds(preds, thresholds):
    binarized = []
    for i, threshold in enumerate(thresholds):
        binary_column = (preds[:, i] >= threshold).astype(int)
        binarized.append(binary_column)
    return np.array(binarized).T


def evaluate_model(train_labels, train_preds, test_labels, test_preds):
    train_auc, train_micro_auc = calculate_auc(train_labels, train_preds)
    test_auc, test_micro_auc = calculate_auc(test_labels, test_preds)
    optimal_thresholds, train_f1_per_tag = optimize_f1_thresholds(train_labels, train_preds)

    train_preds_bin = apply_thresholds(train_preds, optimal_thresholds)
    micro_f1_train = f1_score(train_labels, train_preds_bin, average='micro')

    test_preds_bin = apply_thresholds(test_preds, optimal_thresholds)
    micro_f1_test = f1_score(test_labels, test_preds_bin, average='micro')

    return {'train_micro_auc': train_micro_auc, 'test_micro_auc': test_micro_auc, 'train_micro_f1': micro_f1_train, 'test_micro_f1': micro_f1_test}


def predict_single_text(text, model, sbert_model, thresholds, label_list):
    model.eval()
    with torch.no_grad():
        emb = sbert_model.encode(text, convert_to_tensor=True).unsqueeze(0)  # добавляем batch dimension
        preds = model(emb).cpu().numpy()[0]
        preds_bin = (preds >= thresholds).astype(int)
        predicted_tags = [label for label, pred in zip(label_list, preds_bin) if pred == 1]
    return predicted_tags, preds