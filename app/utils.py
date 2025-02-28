import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set

def extract_skills_from_text(text: str, skill_keywords: Optional[Set[str]] = None) -> List[str]:
    """
    Extract skills from text using keyword matching and patterns.
    
    Args:
        text: Text to extract skills from
        skill_keywords: Optional set of skill keywords to match
        
    Returns:
        list: Extracted skills
    """
    if skill_keywords is None:
        # Default tech skills to look for
        skill_keywords = {
            # Programming Languages
            "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "php", "swift",
            "typescript", "kotlin", "scala", "r", "perl", "bash", "shell", "html", "css",
            
            # Frameworks & Libraries
            "react", "angular", "vue", "django", "flask", "spring", "express", "node.js",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
            ".net", "laravel", "symfony", "jquery", "bootstrap", "tailwind", "flutter", "react native",
            
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis", "cassandra",
            "dynamodb", "firebase", "neo4j", "elasticsearch", "mariadb", "db2",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "gitlab ci",
            "github actions", "ansible", "puppet", "chef", "circleci", "travis", "prometheus",
            "grafana", "istio", "helm", "serverless",
            
            # Data Science & ML
            "machine learning", "deep learning", "nlp", "computer vision", "data mining", 
            "data analysis", "data visualization", "etl", "statistics", "big data", "hadoop",
            "spark", "tableau", "power bi", "qlik", "artificial intelligence", "neural networks",
            
            # Other Tech
            "rest api", "graphql", "websockets", "oauth", "jwt", "microservices", "soa",
            "ci/cd", "test driven development", "agile", "scrum", "kanban", "git", "svn",
            "blockchain", "iot", "embedded systems", "networking", "cybersecurity", "cloud computing",
            
            # Soft Skills (common in tech roles)
            "problem solving", "critical thinking", "communication", "teamwork", "collaboration",
            "leadership", "project management", "time management", "adaptability"
        }
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Find all skill keywords in the text
    found_skills = set()
    for skill in skill_keywords:
        # Use word boundaries to ensure we match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    # Additional pattern matching for skills listed with bullets, commas, etc.
    skill_patterns = [
        r'(?:^|\n)[\s•\-*]+([A-Za-z0-9#\+\.\s]+?)(?:,|$|\n)',  # Bullet points
        r'skills:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',  # After "Skills:" header
        r'technical skills:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',  # After "Technical Skills:" header
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by common separators
            skill_items = re.split(r'[,;•|/]|\s{2,}', match)
            for item in skill_items:
                skill = item.strip()
                if skill and len(skill) > 2:  # Ignore very short items
                    found_skills.add(skill)
    
    return sorted(list(found_skills))

def extract_job_requirements(text: str) -> Dict[str, Any]:
    """
    Extract job requirements from job description text.
    
    Args:
        text: Job description text
        
    Returns:
        dict: Extracted job requirements
    """
    requirements = {
        "required_skills": [],
        "preferred_skills": [],
        "education": [],
        "experience": [],
        "responsibilities": []
    }
    
    # Lower case text for case-insensitive matching
    text_lower = text.lower()
    
    # Extract skills
    required_patterns = [
        r'required skills:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'requirements:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'qualifications:?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    preferred_patterns = [
        r'preferred skills:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'nice to have:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'bonus points:?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    # Extract required skills
    for pattern in required_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by common separators and bullets
            skill_items = re.split(r'[,;•|\n]', match)
            for item in skill_items:
                skill = item.strip()
                if skill and len(skill) > 2 and skill not in requirements["required_skills"]:
                    requirements["required_skills"].append(skill)
    
    # Extract preferred skills
    for pattern in preferred_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by common separators and bullets
            skill_items = re.split(r'[,;•|\n]', match)
            for item in skill_items:
                skill = item.strip()
                if skill and len(skill) > 2 and skill not in requirements["preferred_skills"]:
                    requirements["preferred_skills"].append(skill)
    
    # Extract education requirements
    education_patterns = [
        r'education:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'educational requirements:?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by newlines or bullets
            edu_items = re.split(r'\n|•', match)
            for item in edu_items:
                edu = item.strip()
                if edu and len(edu) > 5 and edu not in requirements["education"]:
                    requirements["education"].append(edu)
    
    # Extract experience requirements
    experience_patterns = [
        r'experience:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'work experience:?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by newlines or bullets
            exp_items = re.split(r'\n|•', match)
            for item in exp_items:
                exp = item.strip()
                if exp and len(exp) > 5 and exp not in requirements["experience"]:
                    requirements["experience"].append(exp)
    
    # Extract responsibilities
    responsibility_patterns = [
        r'responsibilities:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'duties:?\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        r'role:?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
    ]
    
    for pattern in responsibility_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by newlines or bullets
            resp_items = re.split(r'\n|•', match)
            for item in resp_items:
                resp = item.strip()
                if resp and len(resp) > 5 and resp not in requirements["responsibilities"]:
                    requirements["responsibilities"].append(resp)
    
    return requirements

def extract_education_degree_level(education_text: str) -> int:
    """
    Extract education degree level from education text.
    Higher number means higher education level.
    
    Args:
        education_text: Education text to extract level from
        
    Returns:
        int: Education level (0-5)
    """
    education_text = education_text.lower()
    
    # Define education levels
    levels = {
        "phd": 5,
        "doctorate": 5,
        "doctoral": 5,
        "master": 4,
        "msc": 4,
        "ms ": 4,
        "ma ": 4,
        "mba": 4,
        "bachelor": 3,
        "bsc": 3,
        "bs ": 3,
        "ba ": 3,
        "undergraduate": 3,
        "associate": 2,
        "diploma": 2,
        "certificate": 1,
        "high school": 0,
        "ged": 0
    }
    
    highest_level = 0
    for degree, level in levels.items():
        if degree in education_text:
            highest_level = max(highest_level, level)
    
    return highest_level

def calculate_years_of_experience(experience_entries: List[Dict[str, str]]) -> float:
    """
    Calculate total years of experience from experience entries.
    
    Args:
        experience_entries: List of experience entries with date_range
        
    Returns:
        float: Total years of experience
    """
    import re
    from datetime import datetime
    
    total_years = 0.0
    current_year = datetime.now().year
    
    for entry in experience_entries:
        date_range = entry.get('date_range', '')
        
        # Match year ranges (YYYY-YYYY or YYYY-Present)
        match = re.search(r'(\d{4})\s*-\s*((?:\d{4})|present)', date_range, re.IGNORECASE)
        
        if match:
            start_year = int(match.group(1))
            
            if match.group(2).lower() == 'present':
                end_year = current_year
            else:
                end_year = int(match.group(2))
            
            # Add experience duration to total
            total_years += (end_year - start_year)
    
    return total_years

def create_cv_hash(cv_data: Dict[str, Any]) -> str:
    """
    Create a hash of CV data for identification and deduplication.
    
    Args:
        cv_data: CV data to hash
        
    Returns:
        str: Hash of CV data
    """
    # Create a summary of key fields for hashing
    hash_data = {
        "name": cv_data.get('personal_info', {}).get('name', ''),
        "email": cv_data.get('personal_info', {}).get('email', ''),
        "phone": cv_data.get('personal_info', {}).get('phone', ''),
    }
    
    # Add education and experience info if available
    if 'education' in cv_data and cv_data['education']:
        hash_data['education'] = [
            {k: e.get(k, '') for k in ['institution', 'degree']}
            for e in cv_data['education'][:2]  # Use first two entries
        ]
        
    if 'experience' in cv_data and cv_data['experience']:
        hash_data['experience'] = [
            {k: e.get(k, '') for k in ['company', 'job_title']}
            for e in cv_data['experience'][:2]  # Use first two entries
        ]
    
    # Create JSON string and hash it
    json_str = json.dumps(hash_data, sort_keys=True)
    hash_value = hashlib.md5(json_str.encode()).hexdigest()
    
    return hash_value

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        str: Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[^\w\-\.]', '_', filename)
    
    # Limit length to avoid issues with long filenames
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:100] + ext
        
    return sanitized

import os
import shutil

def is_tesseract_installed():
    """Check if Tesseract OCR is installed and available in PATH."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    
    if tesseract_path:
        # Use the custom path from environment variable
        return os.path.exists(tesseract_path)
    else:
        # Check if tesseract is in PATH
        return shutil.which("tesseract") is not None

def merge_cv_data(cv_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple CV data dictionaries into one.
    Useful for combining partial information from multiple sources.
    
    Args:
        cv_data_list: List of CV data dictionaries
        
    Returns:
        dict: Merged CV data
    """
    if not cv_data_list:
        return {}
        
    merged_data = {
        "personal_info": {},
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "document_info": {}
    }
    
    for cv_data in cv_data_list:
        # Merge personal info
        for key, value in cv_data.get("personal_info", {}).items():
            if value and key not in merged_data["personal_info"]:
                merged_data["personal_info"][key] = value
                
        # Merge lists using heuristic deduplication
        for list_key in ["education", "experience", "projects", "certifications"]:
            existing_items = merged_data[list_key]
            new_items = cv_data.get(list_key, [])
            
            for new_item in new_items:
                # Skip if very similar to existing item
                is_duplicate = False
                for existing_item in existing_items:
                    similarity_score = 0
                    total_fields = 0
                    
                    for field in new_item:
                        if field in existing_item:
                            total_fields += 1
                            if str(new_item[field]).lower() == str(existing_item[field]).lower():
                                similarity_score += 1
                                
                    if total_fields > 0 and similarity_score / total_fields > 0.7:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    existing_items.append(new_item)
                    
        # Merge skills with deduplication
        existing_skills = set(merged_data["skills"])
        new_skills = set(cv_data.get("skills", []))
        merged_data["skills"] = sorted(list(existing_skills.union(new_skills)))
        
        # Merge document info
        if "document_info" in cv_data:
            merged_data["document_info"].update(cv_data["document_info"])
            
    return merged_data


if __name__ == "__main__":
    # Example usage
    text = """
    Skills: Python, JavaScript, Machine Learning, Data Analysis
    
    Technical Skills:
    • Programming Languages: Python, JavaScript, Java
    • Frameworks: TensorFlow, PyTorch, React
    • Databases: PostgreSQL, MongoDB
    • Tools: Git, Docker
    
    Work Experience:
    Senior Data Scientist at ABC Corp (2019-Present)
    - Developed machine learning models for customer segmentation
    - Implemented data pipelines using Python and AWS
    
    Data Analyst at XYZ Inc (2017-2019)
    - Performed statistical analysis on large datasets
    - Created interactive dashboards using Tableau
    
    Education:
    Master of Science in Computer Science, Stanford University (2015-2017)
    Bachelor of Science in Statistics, UC Berkeley (2011-2015)
    """
    
    skills = extract_skills_from_text(text)
    print("Extracted Skills:", skills)