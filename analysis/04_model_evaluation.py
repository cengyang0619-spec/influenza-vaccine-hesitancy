import ast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


def classification_metrics(reference, prediction, labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        reference, prediction, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    overall = pd.Series({
        "accuracy": accuracy_score(reference, prediction),
        "cohen_kappa": cohen_kappa_score(reference, prediction, labels=labels),
        "macro_f1": f1.mean(),
    })
    return overall, per_class


def label_set(value):
    if isinstance(value, str):
        return set(ast.literal_eval(value))
    return set(value)


def multilabel_metrics(reference, prediction, labels):
    reference = [label_set(x) for x in reference]
    prediction = [label_set(x) for x in prediction]

    rows = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in gold and label in pred for gold, pred in zip(reference, prediction))
        fp = sum(label not in gold and label in pred for gold, pred in zip(reference, prediction))
        fn = sum(label in gold and label not in pred for gold, pred in zip(reference, prediction))
        tn = len(reference) - tp - fp - fn
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append({
            "label": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "cohen_kappa": cohen_kappa_score(
                [label in x for x in reference],
                [label in x for x in prediction],
            ),
        })

    summary = pd.Series({
        "exact_set_agreement": np.mean([x == y for x, y in zip(reference, prediction)]),
        "micro_precision": total_tp / (total_tp + total_fp),
        "micro_recall": total_tp / (total_tp + total_fn),
        "micro_f1": 2 * total_tp / (2 * total_tp + total_fp + total_fn),
    })
    return summary, pd.DataFrame(rows)


def prevalence_weighted_screening_metrics(frame):
    strata = frame.groupby("predicted_label").agg(
        sample_n=("reference_label", "size"),
        population_n=("population_n", "first"),
    )
    weights = strata["population_n"] / strata["sample_n"]
    sample_weight = frame["predicted_label"].map(weights)
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["reference_label"],
        frame["predicted_label"],
        labels=[True],
        average="binary",
        sample_weight=sample_weight,
        zero_division=0,
    )
    return pd.Series({
        "accuracy": accuracy_score(
            frame["reference_label"],
            frame["predicted_label"],
            sample_weight=sample_weight,
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


stance = pd.read_csv("stance_model_evaluation.csv")
stance_overall, stance_per_class = classification_metrics(
    stance["reference_label"], stance["predicted_label"], [1, 2, 3, 4]
)

stance["reference_hesitant"] = stance["reference_label"].isin([3, 4])
stance["predicted_hesitant"] = stance["predicted_label"].isin([3, 4])
hesitancy_overall, hesitancy_per_class = classification_metrics(
    stance["reference_hesitant"], stance["predicted_hesitant"], [False, True]
)

attribution = pd.read_csv("attribution_model_evaluation.csv")
attribution_labels = pd.read_csv("reason_codebook.csv")["specific_reason"].tolist()
attribution_overall, attribution_per_category = multilabel_metrics(
    attribution["reference_labels"],
    attribution["predicted_labels"],
    attribution_labels,
)

human_reference = pd.read_excel(
    "../data/attribution_analysis_human_annotated_dataset.xlsx",
    sheet_name="reference_data",
)
human_agreement, human_agreement_per_category = multilabel_metrics(
    human_reference["annotator_1_labels"],
    human_reference["annotator_2_labels"],
    attribution_labels,
)

screening = pd.read_csv("screening_validation.csv")
screening_results = screening.groupby("screening_stage").apply(
    prevalence_weighted_screening_metrics,
    include_groups=False,
)
