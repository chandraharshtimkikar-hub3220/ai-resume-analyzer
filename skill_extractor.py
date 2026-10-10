from config import SKILL_VOCAB
from resume_parser import normalize_text


def extract_skills(text):
    """Extract skills using a transparent vocabulary of common technical and soft skills."""
    if text is None:
        return []

    normalized = normalize_text(text).lower()
    found = []
    for skill in SKILL_VOCAB:
        if skill.lower() in normalized and skill not in found:
            found.append(skill)
    return found


def compare_skills(resume_text, job_description):
    """Return the overlap and missing skills between a resume and job description."""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matching_skills": sorted(resume_skills & job_skills),
        "missing_in_resume": sorted(job_skills - resume_skills),
        "resume_only_skills": sorted(resume_skills - job_skills),
    }
