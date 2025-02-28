import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import spacy
from spacy.matcher import Matcher
from langchain.schema import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CVParser")

class CVParser:
    """
    Enhanced CV parser with improved prompts, structured output, and
    fallback mechanisms for robust parsing.
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the CV parser.
        
        Args:
            llm_service: LLM service for enhanced extraction
        """
        # Load NLP model for entity recognition and text processing
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Download model if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
            self.nlp = spacy.load("en_core_web_md")
            
        self.llm_service = llm_service
        self.setup_matchers()
        
    def setup_matchers(self):
        """Setup pattern matchers for various CV sections"""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Pattern for email
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Pattern for phone numbers (improved)
        phone_patterns = [
            [{"SHAPE": {"REGEX": "d{3}[-.]d{3}[-.]d{4}"}}],
            [{"SHAPE": {"REGEX": "[(]d{3}[)][ ]d{3}[-]d{4}"}}],
            [{"TEXT": {"REGEX": "\\+?\\d{1,3}[- ]?\\d{3}[- ]?\\d{3}[- ]?\\d{4}"}}]
        ]
        self.matcher.add("PHONE", phone_patterns)
        
        # Pattern for education section headers (expanded)
        education_patterns = [
            [{"LOWER": "education"}],
            [{"LOWER": "academic"}, {"LOWER": "background"}],
            [{"LOWER": "academic"}, {"LOWER": "qualifications"}],
            [{"LOWER": "educational"}, {"LOWER": "background"}],
            [{"LOWER": "degrees"}],
            [{"LOWER": "academic"}, {"LOWER": "credentials"}],
            [{"LOWER": "academic"}, {"LOWER": "history"}],
            [{"LOWER": "educational"}, {"LOWER": "qualifications"}]
        ]
        self.matcher.add("EDUCATION_SECTION", education_patterns)
        
        # Pattern for experience section headers (expanded)
        experience_patterns = [
            [{"LOWER": "experience"}],
            [{"LOWER": "work"}, {"LOWER": "experience"}],
            [{"LOWER": "professional"}, {"LOWER": "experience"}],
            [{"LOWER": "employment"}, {"LOWER": "history"}],
            [{"LOWER": "work"}, {"LOWER": "history"}],
            [{"LOWER": "career"}, {"LOWER": "history"}],
            [{"LOWER": "professional"}, {"LOWER": "background"}],
            [{"LOWER": "job"}, {"LOWER": "experience"}],
            [{"LOWER": "work"}]
        ]
        self.matcher.add("EXPERIENCE_SECTION", experience_patterns)
        
        # Pattern for skills section headers (expanded)
        skills_patterns = [
            [{"LOWER": "skills"}],
            [{"LOWER": "technical"}, {"LOWER": "skills"}],
            [{"LOWER": "core"}, {"LOWER": "competencies"}],
            [{"LOWER": "key"}, {"LOWER": "skills"}],
            [{"LOWER": "expertise"}],
            [{"LOWER": "professional"}, {"LOWER": "skills"}],
            [{"LOWER": "qualifications"}],
            [{"LOWER": "competencies"}],
            [{"LOWER": "proficiencies"}]
        ]
        self.matcher.add("SKILLS_SECTION", skills_patterns)
        
        # Pattern for projects section headers
        projects_patterns = [
            [{"LOWER": "projects"}],
            [{"LOWER": "academic"}, {"LOWER": "projects"}],
            [{"LOWER": "personal"}, {"LOWER": "projects"}],
            [{"LOWER": "professional"}, {"LOWER": "projects"}],
            [{"LOWER": "key"}, {"LOWER": "projects"}],
            [{"LOWER": "portfolio"}],
            [{"LOWER": "project"}, {"LOWER": "experience"}]
        ]
        self.matcher.add("PROJECTS_SECTION", projects_patterns)
        
        # Pattern for certification section headers
        certification_patterns = [
            [{"LOWER": "certifications"}],
            [{"LOWER": "certificates"}],
            [{"LOWER": "professional"}, {"LOWER": "certifications"}],
            [{"LOWER": "licenses"}, {"TEXT": "&"}, {"LOWER": "certifications"}],
            [{"LOWER": "licenses"}, {"LOWER": "and"}, {"LOWER": "certifications"}],
            [{"LOWER": "credentials"}],
            [{"LOWER": "professional"}, {"LOWER": "development"}]
        ]
        self.matcher.add("CERTIFICATION_SECTION", certification_patterns)
        
        # Pattern for languages section headers
        language_patterns = [
            [{"LOWER": "languages"}],
            [{"LOWER": "language"}, {"LOWER": "skills"}],
            [{"LOWER": "language"}, {"LOWER": "proficiencies"}]
        ]
        self.matcher.add("LANGUAGE_SECTION", language_patterns)
        
        # Pattern for publications section headers
        publication_patterns = [
            [{"LOWER": "publications"}],
            [{"LOWER": "published"}, {"LOWER": "works"}],
            [{"LOWER": "papers"}],
            [{"LOWER": "research"}, {"LOWER": "papers"}],
            [{"LOWER": "articles"}]
        ]
        self.matcher.add("PUBLICATION_SECTION", publication_patterns)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from the CV text.
        
        Args:
            text: Raw CV text
            
        Returns:
            dict: Sections extracted from the CV
        """
        doc = self.nlp(text[:50000])  # Limit to avoid memory issues
        matches = self.matcher(doc)
        
        section_matches = {}
        for match_id, start, end in matches:
            section_name = self.nlp.vocab.strings[match_id]
            if section_name.endswith("_SECTION"):
                section_matches[section_name] = {"start": start, "text": doc[start:end].text}
        
        # Sort matches by their position in the text
        sorted_sections = sorted(section_matches.items(), key=lambda x: x[1]["start"])
        
        # Extract the text between section headers
        sections = {}
        for i, (section_name, section_info) in enumerate(sorted_sections):
            start_idx = doc[section_info["start"]].idx + len(section_info["text"])
            
            # Get end index
            if i < len(sorted_sections) - 1:
                next_section_start = doc[sorted_sections[i+1][1]["start"]].idx
                section_text = text[start_idx:next_section_start].strip()
            else:
                section_text = text[start_idx:].strip()
                
            # Remove the "_SECTION" suffix
            clean_section_name = section_name.replace("_SECTION", "").lower()
            sections[clean_section_name] = section_text
            
        return sections
    
    def _extract_personal_info(self, text: str) -> Dict[str, str]:
        """
        Extract personal information from the CV text.
        
        Args:
            text: Raw CV text
            
        Returns:
            dict: Personal information extracted from the CV
        """
        doc = self.nlp(text[:10000])  # Limit to avoid memory issues
        
        # Extract name (assuming it's at the beginning of the CV or using NER)
        name = text.split('\n')[0].strip()
        
        # Check if name found is likely an actual name using NER
        name_entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name_entities.append(ent.text)
                
        if name_entities and len(name_entities[0].split()) >= 2:
            # Use the first PERSON entity that has at least first and last name
            name = name_entities[0]
        
        # Extract email with improved pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        email = email_matches[0] if email_matches else None
        
        # Extract phone with improved pattern
        phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        phone_matches = re.findall(phone_pattern, text)
        phone = phone_matches[0] if phone_matches else None
        
        # Extract LinkedIn with improved pattern
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=|linkedin\.com/pub/)[A-Za-z0-9_-]+'
        linkedin_matches = re.findall(linkedin_pattern, text)
        linkedin = linkedin_matches[0] if linkedin_matches else None
        
        # Named entity recognition for location (improved)
        locations = []
        address_patterns = [
            r'\b\d+\s+[A-Za-z0-9\s,]+(?:Road|Rd|Street|St|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)(?:\s+[A-Za-z]+,\s+[A-Za-z]+\s+\d{5}(?:-\d{4})?)?',
            r'\b[A-Za-z\s]+,\s+[A-Za-z]{2}\s+\d{5}(?:-\d{4})?'
        ]
        
        # Try to find address patterns
        for pattern in address_patterns:
            address_matches = re.findall(pattern, text)
            if address_matches:
                locations.append(address_matches[0])
                break
                
        # If no address pattern found, use NER
        if not locations:
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    locations.append(ent.text)
                    
        location = locations[0] if locations else None
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "location": location
        }
    
    def parse_cv_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to parse CV text into structured information with enhanced prompting.
        
        Args:
            text: Raw CV text
            
        Returns:
            dict: Structured CV information
        """
        if not self.llm_service:
            raise ValueError("LLM service is required for this method")
            
        system_prompt = """
        You are an expert CV/resume parser AI with extensive experience in talent acquisition and HR.
        Your task is to extract and structure information from a CV/resume document into a comprehensive, 
        standardized format with high accuracy.
        
        Extract the following information, categorized precisely as follows:
        
        1. personal_info: (object)
           - name: Full name of the person
           - email: Email address
           - phone: Phone number in standard format
           - linkedin: LinkedIn profile URL
           - location: Current location/address
           - summary: Brief professional summary or objective if present
           
        2. education: (array of objects)
           For each education entry, include:
           - institution: Name of the educational institution
           - degree: Degree or qualification obtained
           - field: Field of study
           - date_range: Period of study (e.g., "2015-2019")
           - gpa: GPA or academic performance metrics if mentioned
           - location: Location of the institution if specified
           - achievements: Notable academic achievements or honors
           
        3. experience: (array of objects)
           For each work experience entry, include:
           - company: Company or organization name
           - job_title: Job title or position
           - date_range: Period of employment (e.g., "2019-Present")
           - location: Job location
           - responsibilities: Array of responsibilities or achievements in this role
           - skills_used: Specific skills utilized in this role
           
        4. skills: (array of strings)
           Include all skills mentioned, both technical and soft skills
           
        5. projects: (array of objects)
           For each project, include:
           - name: Project name or title
           - description: Brief description of the project
           - date_range: Time period if mentioned
           - technologies: Array of technologies or tools used
           - url: Project URL or repository link if provided
           
        6. certifications: (array of objects)
           For each certification, include:
           - name: Name of certification
           - issuer: Organization that issued the certification
           - date: Date obtained or validity period
           - id: Certification ID if provided
           
        7. languages: (array of objects)
           For each language, include:
           - language: Language name
           - proficiency: Level of proficiency (e.g., "Fluent", "Native", "Intermediate")
           
        8. publications: (array of objects)
           For each publication, include:
           - title: Publication title
           - authors: List of authors
           - date: Publication date
           - publisher: Publisher or journal name
           - url: Link to publication if available
           
        The JSON structure should match these categories exactly. If information for a category is missing,
        include an empty array or object for that category. Be precise and factual in your extraction.
        Do not make assumptions about information that isn't explicitly stated in the CV.
        
        Return the information as a valid JSON object.
        """
        
        user_prompt = f"""
        Here is the text extracted from a CV/resume document. Please parse it according to the instructions:
        
        ```
        {text[:20000]}  # Truncate if too long
        ```
        
        If the text is truncated, focus on extracting information from the available portion.
        """
        
        try:
            response = self.llm_service.chat([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Extract JSON from the response 
            try:
                # Try to extract JSON from code blocks first
                json_text = re.search(r'```(?:json)?\s*(.+?)\s*```', response.content, re.DOTALL)
                if json_text:
                    parsed_data = json.loads(json_text.group(1).strip())
                else:
                    # Try to parse the entire response as JSON
                    parsed_data = json.loads(response.content)
                
                # Add metadata
                parsed_data["metadata"] = {
                    "parser_version": "2.0",
                    "parse_date": datetime.now().isoformat(),
                    "parse_method": "llm",
                    "text_length": len(text)
                }
                
                return parsed_data
                
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                # Fallback to rule-based parsing
                return self.parse_cv_rule_based(text)
                
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}")
            # Fallback to rule-based parsing
            return self.parse_cv_rule_based(text)
    
    def parse_cv_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Use enhanced rule-based approach to parse CV text into structured information.
        
        Args:
            text: Raw CV text
            
        Returns:
            dict: Structured CV information
        """
        # Extract sections
        sections = self._extract_sections(text)
        
        # Extract personal information
        personal_info = self._extract_personal_info(text)
        
        # Extract education (enhanced)
        education = []
        if "education" in sections:
            education_text = sections["education"]
            
            # Split into entries using various delimiters
            education_entries = re.split(r'\n\s*\n|\r\n\s*\r\n|(?<=\n)(?=[A-Z][a-z]+\s+\d{4})', education_text)
            
            for entry in education_entries:
                if len(entry.strip()) > 0:
                    # Try to extract structured information
                    edu_entry = {}
                    
                    # Extract institution
                    institution_patterns = [
                        r'(University|College|Institute|School) of [A-Za-z\s]+',
                        r'([A-Za-z\s]+) University',
                        r'([A-Za-z\s]+) College',
                        r'([A-Za-z\s]+) Institute',
                        r'([A-Za-z\s]+) School'
                    ]
                    
                    for pattern in institution_patterns:
                        institution_match = re.search(pattern, entry)
                        if institution_match:
                            edu_entry["institution"] = institution_match.group(0).strip()
                            break
                    
                    # Extract degree
                    degree_patterns = [
                        r'(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|Ph\.D\.|B\.A\.|M\.A\.|M\.B\.A\.|B\.Eng\.|M\.Eng\.) (?:of|in) [A-Za-z\s]+',
                        r'(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|Ph\.D\.|B\.A\.|M\.A\.|M\.B\.A\.|B\.Eng\.|M\.Eng\.)',
                        r'(Bachelor\'s|Master\'s|Doctoral) degree'
                    ]
                    
                    for pattern in degree_patterns:
                        degree_match = re.search(pattern, entry)
                        if degree_match:
                            edu_entry["degree"] = degree_match.group(0).strip()
                            break
                    
                    # Extract field of study
                    field_patterns = [
                        r'(?:in|of) ([A-Za-z\s]+)(?: Engineering| Science| Arts| Business| Administration| Management)',
                        r'Major:?\s*([A-Za-z\s]+)',
                        r'Field:?\s*([A-Za-z\s]+)'
                    ]
                    
                    for pattern in field_patterns:
                        field_match = re.search(pattern, entry)
                        if field_match:
                            edu_entry["field"] = field_match.group(1).strip()
                            break
                    
                    # Extract date range
                    date_range_match = re.search(r'(\d{4}\s*-\s*(?:\d{4}|Present|Current|Now))', entry, re.IGNORECASE)
                    if date_range_match:
                        edu_entry["date_range"] = date_range_match.group(0).strip()
                    
                    # Extract GPA
                    gpa_match = re.search(r'GPA:?\s*([\d\.]+)(?:/[\d\.]+)?', entry, re.IGNORECASE)
                    if gpa_match:
                        edu_entry["gpa"] = gpa_match.group(1).strip()
                    
                    # If we couldn't extract specific fields, use the raw text
                    if not edu_entry:
                        lines = entry.strip().split('\n')
                        edu_entry["raw_text"] = entry.strip()
                        
                        if lines:
                            # Use first line as degree/institution if nothing else found
                            if "degree" not in edu_entry and "institution" not in edu_entry:
                                if len(lines) >= 2:
                                    edu_entry["degree"] = lines[0].strip()
                                    edu_entry["institution"] = lines[1].strip()
                                elif len(lines) == 1:
                                    edu_entry["institution"] = lines[0].strip()
                    
                    if edu_entry:
                        education.append(edu_entry)
        
        # Extract experience (enhanced)
        experience = []
        if "experience" in sections:
            experience_text = sections["experience"]
            
            # Split into entries
            experience_entries = re.split(r'\n\s*\n|\r\n\s*\r\n|(?<=\n)(?=[A-Z][a-z]+\s+\d{4})', experience_text)
            
            for entry in experience_entries:
                if len(entry.strip()) > 0:
                    # Try to extract structured information
                    exp_entry = {}
                    
                    lines = entry.strip().split('\n')
                    
                    # Extract company and job title (usually in first two lines)
                    if len(lines) >= 2:
                        # Check for patterns like "Job Title at Company" or "Company - Job Title"
                        job_company_pattern = r'(.*?)\s+(?:at|@|,|-)\s+(.*)'
                        job_company_match = re.match(job_company_pattern, lines[0])
                        
                        if job_company_match:
                            exp_entry["job_title"] = job_company_match.group(1).strip()
                            exp_entry["company"] = job_company_match.group(2).strip()
                        else:
                            # Assume first line is job title, second is company
                            exp_entry["job_title"] = lines[0].strip()
                            
                            # Check for date range in the company line
                            company_date_pattern = r'(.*?)(?:\s+\((\d{4}\s*-\s*(?:\d{4}|Present|Current|Now))\))'
                            company_date_match = re.match(company_date_pattern, lines[1])
                            
                            if company_date_match:
                                exp_entry["company"] = company_date_match.group(1).strip()
                                exp_entry["date_range"] = company_date_match.group(2).strip()
                            else:
                                exp_entry["company"] = lines[1].strip()
                    elif len(lines) == 1:
                        exp_entry["raw_text"] = lines[0].strip()
                    
                    # Extract date range if not already found
                    if "date_range" not in exp_entry:
                        date_range_match = re.search(r'(\d{4}\s*-\s*(?:\d{4}|Present|Current|Now))', entry, re.IGNORECASE)
                        if date_range_match:
                            exp_entry["date_range"] = date_range_match.group(0).strip()
                    
                    # Extract responsibilities/achievements
                    responsibilities = []
                    bullet_points = re.findall(r'[•\-\*\+]\s*(.*?)(?=\n[•\-\*\+]|\n\n|$)', entry, re.DOTALL)
                    
                    if bullet_points:
                        responsibilities = [point.strip() for point in bullet_points if point.strip()]
                    
                    if responsibilities:
                        exp_entry["responsibilities"] = responsibilities
                    
                    # Keep raw text for reference
                    exp_entry["raw_text"] = entry.strip()
                    
                    if exp_entry:
                        experience.append(exp_entry)
        
        # Extract skills (enhanced)
        skills = []
        if "skills" in sections:
            skills_text = sections["skills"]
            
            # Try different patterns for skills extraction
            # 1. Bullet points
            bullet_skills = re.findall(r'[•\-\*\+]\s*(.*?)(?=\n[•\-\*\+]|\n\n|$)', skills_text)
            if bullet_skills:
                for skill in bullet_skills:
                    # Check if it's a category with sub-skills
                    sub_skills_match = re.match(r'(.*?):\s*(.*)', skill)
                    if sub_skills_match:
                        sub_skills = [s.strip() for s in sub_skills_match.group(2).split(',')]
                        skills.extend(sub_skills)
                    else:
                        skills.append(skill.strip())
            
            # 2. Comma-separated lists
            elif ',' in skills_text:
                comma_skills = [s.strip() for s in re.split(r',|\n', skills_text) if s.strip()]
                skills.extend(comma_skills)
            
            # 3. Line-separated skills
            else:
                line_skills = [s.strip() for s in skills_text.split('\n') if s.strip()]
                skills.extend(line_skills)
            
            # Clean up skills
            skills = [skill for skill in skills if len(skill) > 1 and not skill.endswith(':')]
        
        # Extract projects (enhanced)
        projects = []
        if "projects" in sections:
            projects_text = sections["projects"]
            
            # Split into entries
            project_entries = re.split(r'\n\s*\n|\r\n\s*\r\n|(?<=\n)(?=[A-Z][a-z]+:)', projects_text)
            
            for entry in project_entries:
                if len(entry.strip()) > 0:
                    # Try to extract structured information
                    project_entry = {}
                    
                    lines = entry.strip().split('\n')
                    
                    # Extract project name (usually first line)
                    if lines:
                        # Check for patterns like "Project Name: Description" or "Project Name - Description"
                        name_desc_pattern = r'(.*?)(?::|–|-)\s*(.*)'
                        name_desc_match = re.match(name_desc_pattern, lines[0])
                        
                        if name_desc_match:
                            project_entry["name"] = name_desc_match.group(1).strip()
                            if name_desc_match.group(2).strip():
                                project_entry["description"] = name_desc_match.group(2).strip()
                        else:
                            project_entry["name"] = lines[0].strip()
                    
                    # Extract description if not already found
                    if "description" not in project_entry and len(lines) > 1:
                        # Use the second line as description
                        project_entry["description"] = lines[1].strip()
                    
                    # Extract technologies
                    tech_patterns = [
                        r'(?:Technologies|Tech Stack|Tools|Stack|Built with)(?:used)?:?\s*(.*)',
                        r'(?:Developed|Implemented|Created|Built)(?:using|with):?\s*(.*)'
                    ]
                    
                    for pattern in tech_patterns:
                        tech_match = re.search(pattern, entry, re.IGNORECASE)
                        if tech_match:
                            techs = [t.strip() for t in tech_match.group(1).split(',')]
                            project_entry["technologies"] = techs
                            break
                    
                    # Extract URL
                    url_match = re.search(r'(?:URL|Link|GitHub):?\s*(https?://\S+)', entry, re.IGNORECASE)
                    if url_match:
                        project_entry["url"] = url_match.group(1).strip()
                    
                    # Keep raw text for reference
                    project_entry["raw_text"] = entry.strip()
                    
                    if project_entry:
                        projects.append(project_entry)
        
        # Extract certifications (enhanced)
        certifications = []
        if "certification" in sections:
            certifications_text = sections["certification"]
            
            # Split into entries
            cert_entries = re.split(r'\n\s*\n|\r\n\s*\r\n|(?<=\n)(?=[A-Z])|[•\-\*\+]\s*', certifications_text)
            
            for entry in cert_entries:
                entry = entry.strip()
                if entry:
                    # Try to extract structured information
                    cert_entry = {"name": entry}
                    
                    # Check for issuer
                    issuer_patterns = [
                        r'(?:issued by|from|by|through)\s+(.*)',
                        r'(.*?)(?:certification|certificate)'
                    ]
                    
                    for pattern in issuer_patterns:
                        issuer_match = re.search(pattern, entry, re.IGNORECASE)
                        if issuer_match:
                            cert_entry["issuer"] = issuer_match.group(1).strip()
                            break
                    
                    # Extract date
                    date_match = re.search(r'(?:issued|completed|obtained|earned)(?:in|on)?\s+([A-Za-z]+\s+\d{4}|\d{4})', entry, re.IGNORECASE)
                    if date_match:
                        cert_entry["date"] = date_match.group(1).strip()
                    
                    certifications.append(cert_entry)
        
        # Extract languages
        languages = []
        if "language" in sections:
            languages_text = sections["language"]
            
            # Try different patterns for languages extraction
            language_entries = re.split(r'[,\n•\-\*\+]', languages_text)
            
            for entry in language_entries:
                entry = entry.strip()
                if entry:
                    # Try to extract language and proficiency
                    language_entry = {}
                    
                    # Check for patterns like "Language (Proficiency)" or "Language: Proficiency"
                    prof_pattern = r'(.*?)(?:\(([^)]+)\)|:\s*([^,\n]+))'
                    prof_match = re.match(prof_pattern, entry)
                    
                    if prof_match:
                        language_entry["language"] = prof_match.group(1).strip()
                        language_entry["proficiency"] = prof_match.group(2) or prof_match.group(3).strip()
                    else:
                        language_entry["language"] = entry
                    
                    if language_entry:
                        languages.append(language_entry)
        
        # Extract publications
        publications = []
        if "publication" in sections:
            publications_text = sections["publication"]
            
            # Split into entries
            pub_entries = re.split(r'\n\s*\n|\r\n\s*\r\n|(?<=\n)(?=[A-Z])|[•\-\*\+]\s*', publications_text)
            
            for entry in pub_entries:
                entry = entry.strip()
                if entry:
                    # Try to extract structured information
                    pub_entry = {"title": entry}
                    
                    # Extract authors
                    authors_match = re.search(r'(?:with|authored by|co-authored with)\s+(.*)', entry, re.IGNORECASE)
                    if authors_match:
                        pub_entry["authors"] = authors_match.group(1).strip()
                    
                    # Extract date
                    date_match = re.search(r'(?:published|published in|published on|in)\s+([A-Za-z]+\s+\d{4}|\d{4})', entry, re.IGNORECASE)
                    if date_match:
                        pub_entry["date"] = date_match.group(1).strip()
                    
                    # Extract publisher/journal
                    publisher_match = re.search(r'(?:in|by)\s+((?:The\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Journal|Press|Conference|Symposium)?)', entry)
                    if publisher_match:
                        pub_entry["publisher"] = publisher_match.group(1).strip()
                    
                    publications.append(pub_entry)
        
        # Combine all extracted information
        structured_cv = {
            "personal_info": personal_info,
            "education": education,
            "experience": experience,
            "skills": skills,
            "projects": projects,
            "certifications": certifications,
            "languages": languages,
            "publications": publications,
            "metadata": {
                "parser_version": "2.0",
                "parse_date": datetime.now().isoformat(),
                "parse_method": "rule_based",
                "text_length": len(text)
            }
        }
        
        return structured_cv
    
    def parse_cv(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Parse CV text into structured information using either LLM or rule-based approach.
        
        Args:
            text: Raw CV text
            use_llm: Whether to use LLM for parsing
            
        Returns:
            dict: Structured CV information
        """
        try:
            if use_llm and self.llm_service:
                logger.info("Parsing CV with LLM")
                return self.parse_cv_with_llm(text)
            else:
                logger.info("Parsing CV with rule-based approach")
                return self.parse_cv_rule_based(text)
        except Exception as e:
            logger.error(f"Error parsing CV: {str(e)}")
            # Return a minimal structure in case of error
            return {
                "personal_info": self._extract_personal_info(text),
                "education": [],
                "experience": [],
                "skills": [],
                "projects": [],
                "certifications": [],
                "languages": [],
                "publications": [],
                "metadata": {
                    "parser_version": "2.0",
                    "parse_date": datetime.now().isoformat(),
                    "parse_method": "error_fallback",
                    "error": str(e),
                    "text_length": len(text)
                }
            }
            
    def batch_parse(self, documents: List[dict], use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Parse a batch of CV documents with enhanced error handling.
        
        Args:
            documents: List of document information dictionaries
            use_llm: Whether to use LLM for parsing
            
        Returns:
            list: List of parsed CV information dictionaries
        """
        parsed_cvs = []
        
        for doc in documents:
            try:
                logger.info(f"Parsing CV from document: {doc.get('filename', 'unknown')}")
                text = doc.get("raw_text", "")
                
                if not text:
                    logger.warning(f"Empty text for document: {doc.get('filename', 'unknown')}")
                    continue
                
                parsed_cv = self.parse_cv(text, use_llm)
                
                # Add document metadata
                parsed_cv["document_info"] = {
                    "filename": doc.get("filename"),
                    "is_scanned": doc.get("is_scanned", False),
                    "text_quality": doc.get("text_quality", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "extension": doc.get("extension", ""),
                    "ocr_applied": doc.get("ocr_applied", False)
                }
                
                parsed_cvs.append(parsed_cv)
                
            except Exception as e:
                logger.error(f"Error parsing CV {doc.get('filename')}: {str(e)}")
                # Add error entry
                parsed_cvs.append({
                    "personal_info": {"name": f"Error parsing {doc.get('filename', 'unknown')}"},
                    "document_info": {
                        "filename": doc.get("filename"),
                        "error": str(e)
                    },
                    "metadata": {
                        "parser_version": "2.0",
                        "parse_date": datetime.now().isoformat(),
                        "parse_method": "error",
                        "error": str(e)
                    }
                })
                
        return parsed_cvs
    
    def save_parsed_cvs(self, parsed_cvs: List[Dict[str, Any]], output_dir: Union[str, Path]):
        """
        Save the parsed CVs to the output directory.
        
        Args:
            parsed_cvs: List of parsed CV information dictionaries
            output_dir: Directory to save the parsed data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for cv in parsed_cvs:
            filename = cv.get("document_info", {}).get("filename", f"cv_{cv.get('personal_info', {}).get('name', 'unknown')}.json")
            filename = re.sub(r'[^\w\-\.]', '_', filename)  # Sanitize filename
            output_file = output_dir / f"parsed_{Path(filename).stem}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cv, f, ensure_ascii=False, indent=2)
        
        # Create a summary CSV
        self._save_summary_csv(parsed_cvs, output_dir / "cv_summary.csv")
    
    def _save_summary_csv(self, parsed_cvs: List[Dict[str, Any]], output_file: Path):
        """
        Save a summary CSV of the parsed CVs.
        
        Args:
            parsed_cvs: List of parsed CV information dictionaries
            output_file: Output CSV file path
        """
        try:
            import pandas as pd
            
            # Prepare summary data
            summary_data = []
            
            for cv in parsed_cvs:
                personal_info = cv.get("personal_info", {})
                education = cv.get("education", [])
                experience = cv.get("experience", [])
                skills = cv.get("skills", [])
                document_info = cv.get("document_info", {})
                
                # Calculate experience duration
                total_years = 0
                for exp in experience:
                    date_range = exp.get("date_range", "")
                    if "-" in date_range:
                        parts = date_range.split("-")
                        try:
                            start_year = int(re.search(r'\d{4}', parts[0]).group(0))
                            end_year = datetime.now().year if ("present" in parts[1].lower() or 
                                                               "current" in parts[1].lower() or 
                                                               "now" in parts[1].lower()) else int(re.search(r'\d{4}', parts[1]).group(0))
                            total_years += (end_year - start_year)
                        except (AttributeError, ValueError):
                            pass
                
                # Get highest education
                highest_edu = ""
                for edu in education:
                    degree = edu.get("degree", "")
                    if degree:
                        if "phd" in degree.lower() or "doctor" in degree.lower():
                            highest_edu = degree
                            break
                        elif "master" in degree.lower() and not "phd" in highest_edu.lower():
                            highest_edu = degree
                        elif "bachelor" in degree.lower() and not ("master" in highest_edu.lower() or "phd" in highest_edu.lower()):
                            highest_edu = degree
                
                summary_data.append({
                    "Name": personal_info.get("name", "Unknown"),
                    "Email": personal_info.get("email", ""),
                    "Phone": personal_info.get("phone", ""),
                    "Location": personal_info.get("location", ""),
                    "Highest Education": highest_edu,
                    "Years of Experience": total_years,
                    "Skills Count": len(skills),
                    "Top Skills": ", ".join(skills[:5]) if skills else "",
                    "Filename": document_info.get("filename", ""),
                    "Parse Method": cv.get("metadata", {}).get("parse_method", "")
                })
            
            # Create DataFrame and save CSV
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False)
            
        except ImportError:
            logger.warning("pandas is required for saving summary CSV. Skipped.")
        except Exception as e:
            logger.error(f"Error saving summary CSV: {str(e)}")