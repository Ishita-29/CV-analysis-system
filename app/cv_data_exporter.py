import re
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_exporter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CVDataExporter")

class CVDataExporter:
    """
    A class for exporting parsed CV data to Excel format with detailed
    structured sheets for analytics and comparison.
    """
    
    def __init__(self):
        """Initialize the CV Data Exporter"""
        pass
    
    def export_to_excel(self, parsed_cvs: Dict[str, Dict[str, Any]], output_file: Union[str, Path],
                       include_sheets: Optional[List[str]] = None) -> Path:
        """
        Export parsed CV data to a comprehensive Excel file with multiple sheets.
        
        Args:
            parsed_cvs: Dictionary mapping CV IDs to parsed CV data
            output_file: Path to the output Excel file
            include_sheets: List of sheets to include (default: all)
            
        Returns:
            Path: Path to the created Excel file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Simplify the data for easier reading
        simplified_data = {}
        for cv_id, cv in parsed_cvs.items():
            # Extract only essential information
            personal_info = cv.get("personal_info", {})
            name = personal_info.get("name", "Unknown")
            
            simplified_data[cv_id] = {
                "name": name,
                "contact": {
                    "email": personal_info.get("email", ""),
                    "phone": personal_info.get("phone", ""),
                    "location": personal_info.get("location", "")
                },
                "education": [
                    {
                        "degree": edu.get("degree", ""),
                        "institution": edu.get("institution", ""),
                        "date": edu.get("date_range", "")
                    } for edu in cv.get("education", [])
                ],
                "experience": [
                    {
                        "title": exp.get("job_title", ""),
                        "company": exp.get("company", ""),
                        "date": exp.get("date_range", "")
                    } for exp in cv.get("experience", [])
                ],
                "skills": cv.get("skills", [])
            }
        
        # Create a Pandas Excel writer
        try:
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            workbook = writer.book
            
            # Define formats for a clean look
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            # Create Summary sheet
            summary_data = []
            for cv_id, cv in simplified_data.items():
                summary_data.append({
                    "Name": cv["name"],
                    "Email": cv["contact"]["email"],
                    "Phone": cv["contact"]["phone"],
                    "Location": cv["contact"]["location"],
                    "Skills Count": len(cv["skills"]),
                    "Education Count": len(cv["education"]),
                    "Experience Count": len(cv["experience"])
                })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # Format the summary sheet
            worksheet = writer.sheets['Summary']
            for col_num, value in enumerate(pd.DataFrame(summary_data).columns.values):
                worksheet.write(0, col_num, value, header_format)
                # Set reasonable column width
                worksheet.set_column(col_num, col_num, 20)
            
            # Skills sheet - simplified
            skills_data = []
            for cv_id, cv in simplified_data.items():
                for skill in cv["skills"]:
                    # Skip None values
                    if skill is None:
                        continue
                    
                    # Safely identify skill type
                    skill_lower = self._safe_lower(skill)
                    
                    # Simple categorization
                    skill_type = "Other"
                    if any(tech in skill_lower for tech in ['python', 'java', 'javascript', 'sql', 'html', 'css', 'programming']):
                        skill_type = "Technical"
                    elif any(soft in skill_lower for soft in ['communication', 'leadership', 'teamwork', 'management']):
                        skill_type = "Soft"
                    
                    skills_data.append({
                        "Name": cv["name"],
                        "Skill": skill,
                        "Type": skill_type
                    })
            
            if skills_data:
                pd.DataFrame(skills_data).to_excel(writer, sheet_name="Skills", index=False)
                
                # Format skills sheet
                worksheet = writer.sheets['Skills']
                for col_num, value in enumerate(pd.DataFrame(skills_data).columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 20)
            
            # Education sheet - simplified
            education_data = []
            for cv_id, cv in simplified_data.items():
                for edu in cv["education"]:
                    education_data.append({
                        "Name": cv["name"],
                        "Degree": edu["degree"],
                        "Institution": edu["institution"],
                        "Date": edu["date"]
                    })
            
            if education_data:
                pd.DataFrame(education_data).to_excel(writer, sheet_name="Education", index=False)
                
                # Format education sheet
                worksheet = writer.sheets['Education']
                for col_num, value in enumerate(pd.DataFrame(education_data).columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 20)
            
            # Experience sheet - simplified
            experience_data = []
            for cv_id, cv in simplified_data.items():
                for exp in cv["experience"]:
                    experience_data.append({
                        "Name": cv["name"],
                        "Title": exp["title"],
                        "Company": exp["company"],
                        "Date": exp["date"]
                    })
            
            if experience_data:
                pd.DataFrame(experience_data).to_excel(writer, sheet_name="Experience", index=False)
                
                # Format experience sheet
                worksheet = writer.sheets['Experience']
                for col_num, value in enumerate(pd.DataFrame(experience_data).columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 20)
            
            # Save the Excel file
            writer.close()
            
            logger.info(f"Excel export completed: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            logger.error(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}")
            raise
    
    def _safe_lower(self, value):
        """Safely convert a value to lowercase, handling None values."""
        if value is None:
            return ""
        elif isinstance(value, str):
            return value.lower()
        else:
            return str(value).lower()
    
    def _calculate_experience_years(self, experience: List[Dict[str, Any]]) -> float:
        """Calculate total years of experience from experience entries"""
        total_years = 0.0
        current_year = datetime.now().year
        
        for exp in experience:
            date_range = exp.get('date_range', '')
            
            # Skip if no date range
            if not date_range:
                continue
                
            # Match year ranges (YYYY-YYYY or YYYY-Present)
            match = re.search(r'(\d{4})\s*-\s*((?:\d{4})|present|current|now)', date_range, re.IGNORECASE)
            
            if match:
                start_year = int(match.group(1))
                
                if match.group(2).lower() in ('present', 'current', 'now'):
                    end_year = current_year
                else:
                    try:
                        end_year = int(match.group(2))
                    except ValueError:
                        continue
                
                # Add experience duration to total
                total_years += (end_year - start_year)
        
        return round(total_years, 1)
    
    def _get_highest_degree(self, education: List[Dict[str, Any]]) -> str:
        """Get highest degree from education entries"""
        highest_level = 0
        highest_degree = ""
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            
            # Determine education level
            level = 0
            if degree and ('phd' in degree or 'doctor' in degree):
                level = 4
            elif degree and 'master' in degree:
                level = 3
            elif degree and 'bachelor' in degree:
                level = 2
            elif degree and ('associate' in degree or 'diploma' in degree):
                level = 1
            
            if level > highest_level:
                highest_level = level
                highest_degree = edu.get('degree', '')
        
        return highest_degree
    
    def export_to_csv(self, parsed_cvs: Dict[str, Dict[str, Any]], 
                     output_dir: Union[str, Path],
                     file_prefix: str = "cv_data_") -> Dict[str, Path]:
        """
        Export parsed CV data to a single CSV file.
        
        Args:
            parsed_cvs: Dictionary mapping CV IDs to parsed CV data
            output_dir: Directory to save the CSV files
            file_prefix: Prefix for the CSV filenames
            
        Returns:
            dict: Dictionary mapping sheet names to CSV file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a single CSV file
        output_file = output_dir / f"{file_prefix}all_cvs.csv"
        
        try:
            # Prepare data rows
            csv_data = []
            for cv_id, cv in parsed_cvs.items():
                personal_info = cv.get("personal_info", {})
                education = cv.get("education", [])
                experience = cv.get("experience", [])
                skills = cv.get("skills", [])
                
                # Extract education and experience details
                edu_details = []
                for edu in education:
                    if edu.get("degree") and edu.get("institution"):
                        edu_details.append(f"{edu.get('degree')} at {edu.get('institution')}")
                
                exp_details = []
                for exp in experience:
                    if exp.get("job_title") and exp.get("company"):
                        exp_details.append(f"{exp.get('job_title')} at {exp.get('company')}")
                
                # Create a row for this CV
                row = {
                    "Name": personal_info.get("name", "Unknown"),
                    "Email": personal_info.get("email", ""),
                    "Phone": personal_info.get("phone", ""),
                    "Location": personal_info.get("location", ""),
                    "Skills": ", ".join([s for s in skills if s is not None]),
                    "Education": ", ".join(edu_details),
                    "Experience": ", ".join(exp_details)
                }
                csv_data.append(row)
            
            # Write to CSV
            pd.DataFrame(csv_data).to_csv(output_file, index=False)
            
            logger.info(f"CSV export completed to: {output_file}")
            return {"all_cvs": output_file}
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise