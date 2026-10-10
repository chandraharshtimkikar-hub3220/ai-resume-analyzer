import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_VECTORIZER_PATH,
    LABEL_COLUMN_ENV,
    TEXT_COLUMN_ENV,
    ensure_project_dirs,
)
from resume_parser import normalize_text


def build_synthetic_dataset():
    """Provide a tiny fallback dataset for testing the ML pipeline when no real data is available."""
    records = [
        {"resume_text": "Python developer with SQL, pandas, numpy, machine learning, scikit-learn, Git, teamwork.", "category": "Data Science"},
        {"resume_text": "Software engineer experienced in Java, C++, data structures, algorithms, Git, communication.", "category": "Software Engineering"},
        {"resume_text": "Machine learning engineer with NLP, deep learning, TensorFlow, Python, model evaluation, data analysis.", "category": "Machine Learning"},
        {"resume_text": "Frontend developer with JavaScript, HTML, CSS, React, UI design, teamwork, communication.", "category": "Web Development"},
        {"resume_text": "Data analyst with SQL, Excel, Power BI, data visualization, business insights, Python.", "category": "Data Analytics"},
        {"resume_text": "Cybersecurity analyst with network security, Linux, Python, incident response, communication.", "category": "Cybersecurity"},
        {"resume_text": "Android developer with Java, Kotlin, Android Studio, APIs, testing, teamwork.", "category": "Mobile Development"},
        {"resume_text": "Research assistant in NLP, statistics, Python, data analysis, communication, literature review.", "category": "Research"},
    ]
    return pd.DataFrame(records)


def auto_detect_columns(df):
    """Infer likely text and label columns in a resume dataset without assuming names."""
    label_candidates = [
        "category",
        "Category",
        "job_category",
        "Job_Category",
        "label",
        "Label",
        "class",
        "Class",
        "resume_category",
        "Resume_Category",
    ]
    text_candidates = [
        "resume_text",
        "Resume",
        "resume",
        "text",
        "Description",
        "description",
        "content",
        "resume_content",
        "resume_summary",
        "summary",
        "job_description",
    ]

    text_column = next((col for col in text_candidates if col in df.columns), None)
    label_column = next((col for col in label_candidates if col in df.columns), None)

    if text_column is None:
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ["resume", "description", "text", "summary"]):
                text_column = col
                break
    if label_column is None:
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ["category", "label", "class"]):
                label_column = col
                break
    return text_column, label_column


def load_resume_dataset(dataset_path=None, text_column=None, label_column=None):
    """Load the real dataset if it exists; otherwise fall back to a synthetic dataset for testing."""
    dataset_path = dataset_path or os.getenv("RESUME_DATASET_PATH", str(DEFAULT_DATASET_PATH))
    path = Path(dataset_path)

    if path.exists() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
        df = pd.read_csv(path)
        if text_column is None or label_column is None:
            detected_text, detected_label = auto_detect_columns(df)
            text_column = text_column or detected_text
            label_column = label_column or detected_label

        if text_column is None or label_column is None:
            raise ValueError(
                "Could not detect the resume text and category columns. "
                "Please provide --text-column and --label-column explicitly."
            )
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(
                f"Columns not found in dataset: text_column={text_column}, label_column={label_column}. "
                f"Available columns: {list(df.columns)}"
            )
        return df[[text_column, label_column]].rename(columns={text_column: "resume_text", label_column: "category"})

    return build_synthetic_dataset()


def train_and_evaluate_model(
    dataset_path=None,
    text_column=None,
    label_column=None,
    model_path=None,
    vectorizer_path=None,
    labels_path=None,
    metrics_path=None,
):
    """Train a text classifier on resume categories and save the ML assets."""
    ensure_project_dirs()
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    vectorizer_path = Path(vectorizer_path or DEFAULT_VECTORIZER_PATH)
    labels_path = Path(labels_path or DEFAULT_LABELS_PATH)
    metrics_path = Path(metrics_path or DEFAULT_METRICS_PATH)

    df = load_resume_dataset(dataset_path=dataset_path, text_column=text_column, label_column=label_column)
    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid resume dataset.")

    df = df.dropna(subset=["resume_text", "category"]).copy()
    df["resume_text"] = df["resume_text"].apply(normalize_text)
    df["category"] = df["category"].astype(str)

    if df.empty:
        raise ValueError("The dataset contains no usable resume text or category labels.")
    if df["category"].nunique() < 2:
        raise ValueError("At least two category labels are required for model training.")

    X = df["resume_text"]
    y = df["category"]

    if len(df) >= 10 and y.nunique() >= 2:
        stratify = y
    else:
        stratify = None
        print("Warning: dataset is too small for a reliable stratified split; using a standard split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25 if len(df) >= 8 else 0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": sorted(y.unique().tolist()),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    joblib.dump(pipeline.named_steps["classifier"], model_path)
    joblib.dump(sorted(y.unique().tolist()), labels_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision (macro): {metrics['precision_macro']}")
    print(f"Recall (macro): {metrics['recall_macro']}")
    print(f"F1 score (macro): {metrics['f1_macro']}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return pipeline, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the resume category classifier.")
    parser.add_argument("--dataset-path", default=os.getenv("RESUME_DATASET_PATH", str(DEFAULT_DATASET_PATH)))
    parser.add_argument("--text-column", default=os.getenv(TEXT_COLUMN_ENV))
    parser.add_argument("--label-column", default=os.getenv(LABEL_COLUMN_ENV))
    args = parser.parse_args()

    train_and_evaluate_model(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )
