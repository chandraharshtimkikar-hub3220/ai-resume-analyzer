from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATASET_PATH = DATA_DIR / "resume_dataset.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "resume_classifier.joblib"
DEFAULT_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
DEFAULT_LABELS_PATH = MODELS_DIR / "class_labels.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

SKILL_VOCAB = [
    "Python",
    "SQL",
    "Java",
    "C++",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Data Analysis",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "Git",
    "Communication",
    "Teamwork",
    "Statistics",
    "Excel",
    "Power BI",
    "Tableau",
    "Data Visualization",
    "Computer Vision",
    "Artificial Intelligence",
    "R",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "C#",
    "JavaScript",
    "HTML",
    "CSS",
    "React",
    "Django",
    "Flask",
    "AWS",
    "Docker",
    "Kubernetes",
]

DATASET_PATH_ENV = "RESUME_DATASET_PATH"
TEXT_COLUMN_ENV = "RESUME_TEXT_COLUMN"
LABEL_COLUMN_ENV = "RESUME_LABEL_COLUMN"


def ensure_project_dirs():
    for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
