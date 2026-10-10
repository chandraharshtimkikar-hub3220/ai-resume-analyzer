from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import normalize_text


def compute_similarity_score(resume_text, job_description):
    """Compare resume and job description using TF-IDF plus cosine similarity."""
    resume_text = normalize_text(resume_text)
    job_description = normalize_text(job_description)

    if not resume_text or not job_description:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)
