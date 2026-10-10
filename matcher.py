import os
import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import normalize_text


def compute_similarity_score(resume_text, job_description):
    """Compare a resume and job description using TF-IDF and cosine similarity."""
    resume_clean = normalize_text(resume_text)
    job_clean = normalize_text(job_description)

    if not resume_clean or not job_clean:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)


def generate_skill_summary(resume_text, job_description):
    """Helper for matching and gap analysis."""
    from skill_extractor import extract_skills, compare_skills

    return compare_skills(resume_text, job_description)
