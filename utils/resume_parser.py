import re

def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'\b\d{10}\b', text)
    return match.group() if match else None

def extract_skills(text, skill_list):
    skills_found = []
    for skill in skill_list:
        if skill.lower() in text.lower():
            skills_found.append(skill)
    return skills_found

def extract_years(text):
    match=re.findall(r'\b(19|20\d{2})\b',text)
    return match.group() if match else None