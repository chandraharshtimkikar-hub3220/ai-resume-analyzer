import re

from config import SKILL_VOCAB
from resume_parser import normalize_text


def extract_skills(text):
    """Extract skills from text using a transparent, editable vocabulary."""
    cleaned = normalize_text(text).lower()
    skills_found = []

    for skill in SKILL_VOCAB:
        skill_lower = skill.lower()
        if skill_lower in cleaned:
            skills_found.append(skill)

    ordered = []
    seen = set()
    for skill in SKILL_VOCAB:
        if skill in skills_found and skill not in seen:
            ordered.append(skill)
            seen.add(skill)
    return ordered


def compare_skills(resume_text, job_description):
    """Compare resume skills vs job requirements and return matching and missing skills."""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    matching = sorted(resume_skills & job_skills)
    missing_in_resume = sorted(job_skills - resume_skills)
    resume_only = sorted(resume_skills - job_skills)

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matching_skills": matching,
        "missing_in_resume": missing_in_resume,
        "resume_only_skills": resume_only,
    }
