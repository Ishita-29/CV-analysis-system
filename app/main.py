import os
import json
import time
import logging
import tempfile
import platform
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

# Import enhanced components
from document_processor import DocumentProcessor
from cv_parser import CVParser
from llm_service import LLMService
from chatbot import CVChatbot  # Use enhanced chatbot
from cv_data_exporter import CVDataExporter
# from cv_data_exporter import DataExporter as CVDataExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_analysis_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CVAnalysisSystem")

# Load environment variables
load_dotenv()

# Application title and configuration
st.set_page_config(
    page_title="Advanced CV Analysis System",
    page_icon="📄",
    layout="wide"
)

# Check if Tesseract is installed
def is_tesseract_installed():
    """Check if Tesseract OCR is installed and available in PATH."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    
    if tesseract_path:
        # Use the custom path from environment variable
        return os.path.exists(tesseract_path)
    else:
        # Check if tesseract is in PATH
        return shutil.which("tesseract") is not None

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'processed_cvs' not in st.session_state:
        st.session_state.processed_cvs = {}
        
    if 'parsed_cvs' not in st.session_state:
        st.session_state.parsed_cvs = {}
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'context_id' not in st.session_state:
        st.session_state.context_id = None
        
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
        
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        
    if 'tab_selection' not in st.session_state:
        st.session_state.tab_selection = "Upload & Process"
        
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
        
    if 'export_status' not in st.session_state:
        st.session_state.export_status = None
        
    if 'selected_cvs' not in st.session_state:
        st.session_state.selected_cvs = []
        
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
        
    if 'data_exporter' not in st.session_state:
        st.session_state.data_exporter = CVDataExporter()
        
@st.cache_resource
def get_llm_service():
    """Initialize and cache the LLM service"""
    provider = os.getenv("LLM_PROVIDER", "groq")
    model_name = os.getenv("LLM_MODEL", "llama-3-8b-8192")
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    
    # Get fallback providers from environment
    fallback_providers = []
    fallback_config = os.getenv("FALLBACK_PROVIDERS")
    if fallback_config:
        try:
            fallback_providers = json.loads(fallback_config)
        except json.JSONDecodeError:
            logger.error("Error parsing FALLBACK_PROVIDERS environment variable")
    
    # Get rate limit settings
    requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
    max_retries = int(os.getenv("MAX_RETRIES", "5"))
    
    logger.info(f"Initializing LLM service with provider: {provider}, model: {model_name}")
    
    return LLMService(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        max_retries=max_retries,
        requests_per_minute=requests_per_minute,
        fallback_providers=fallback_providers
    )

def process_uploaded_files(uploaded_files, use_ocr=True, ocr_language="eng"):
    """Process uploaded CV files with enhanced document processor"""
    with st.spinner("Processing CV documents..."):
        # Check if Tesseract is installed if OCR is requested
        ocr_enabled = use_ocr
        if use_ocr and not is_tesseract_installed():
            st.warning("""
            Tesseract OCR is not installed or not found in PATH. 
            Processing will continue without OCR, which might affect text extraction from scanned documents.
            
            See the Installation Guide in the README for instructions on installing Tesseract.
            """)
            ocr_enabled = False
        
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            # Save uploaded files to temporary directory
            for uploaded_file in uploaded_files:
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                
            # Process the documents with enhanced processor
            processor = DocumentProcessor(ocr_enabled=ocr_enabled, ocr_lang=ocr_language)
            processed_documents = []
            
            progress_bar = st.progress(0)
            st.session_state.processing_status = ""
            
            for i, file_path in enumerate(file_paths):
                try:
                    status_msg = f"Processing {file_path.name}..."
                    st.session_state.processing_status = status_msg
                    logger.info(status_msg)
                    
                    doc_info = processor.process_document(file_path)
                    processed_documents.append(doc_info)
                    
                    # Log document quality
                    quality_msg = f"Document quality: {doc_info.get('text_quality', 'unknown')}"
                    if doc_info.get('is_scanned', False):
                        quality_msg += " (Scanned document detected)"
                    if doc_info.get('ocr_applied', False):
                        quality_msg += " (OCR applied)"
                    
                    logger.info(quality_msg)
                    st.session_state.processing_status += f"\n{quality_msg}"
                    
                    # Update session state
                    st.session_state.processed_cvs[file_path.name] = doc_info
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {str(e)}"
                    logger.error(error_msg)
                    st.session_state.processing_status += f"\n❌ {error_msg}"
                    
                progress_bar.progress((i + 1) / len(file_paths))
                
        # Export processing summary
        try:
            summary_file = Path("data/processed/document_summary.csv")
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            processor.save_processed_documents(processed_documents, summary_file.parent)
            st.session_state.processing_status += f"\n✅ Processing summary saved to {summary_file}"
        except Exception as e:
            logger.error(f"Error saving processing summary: {str(e)}")
                
    return processed_documents

def parse_processed_documents(processed_documents, use_llm=True, job_position=None):
    """Parse processed CV documents to extract structured information with enhanced parser"""
    with st.spinner("Parsing CV information..."):
        llm_service = get_llm_service()
        parser = CVParser(llm_service=llm_service)
        
        parsed_cvs = []
        progress_bar = st.progress(0)
        st.session_state.processing_status = "Parsing documents..."
        
        for i, doc in enumerate(processed_documents):
            try:
                status_msg = f"Parsing {doc['filename']}..."
                st.session_state.processing_status += f"\n{status_msg}"
                logger.info(status_msg)
                
                text = doc.get("raw_text", "")
                parsed_cv = parser.parse_cv(text, use_llm)
                
                # Add document metadata
                parsed_cv["document_info"] = {
                    "filename": doc.get("filename"),
                    "is_scanned": doc.get("is_scanned", False),
                    "text_quality": doc.get("text_quality", "unknown"),
                    "ocr_applied": doc.get("ocr_applied", False),
                    "file_size": doc.get("file_size", 0),
                    "extension": doc.get("extension", "")
                }
                
                # Add job position if provided
                if job_position:
                    parsed_cv["job_position"] = job_position
                
                parsed_cvs.append(parsed_cv)
                
                # Update session state
                cv_id = doc['filename']
                st.session_state.parsed_cvs[cv_id] = parsed_cv
                
            except Exception as e:
                error_msg = f"Error parsing CV {doc.get('filename')}: {str(e)}"
                logger.error(error_msg)
                st.session_state.processing_status += f"\n❌ {error_msg}"
                
            progress_bar.progress((i + 1) / len(processed_documents))
            
        # Initialize or update chatbot with parsed CVs
        if st.session_state.chatbot is None:
            st.session_state.chatbot = CVChatbot(llm_service, st.session_state.parsed_cvs)
        else:
            for cv_id, cv_data in st.session_state.parsed_cvs.items():
                st.session_state.chatbot.add_cv(cv_id, cv_data)
                
        # Save parsed CVs
        try:
            output_dir = Path("data/parsed")
            output_dir.mkdir(parents=True, exist_ok=True)
            parser.save_parsed_cvs(parsed_cvs, output_dir)
            st.session_state.processing_status += f"\n✅ Parsed CVs saved to {output_dir}"
            
            # Export to Excel
            try:
                excel_file = output_dir / "cv_data.xlsx"
                st.session_state.data_exporter.export_to_excel(st.session_state.parsed_cvs, excel_file)
                st.session_state.processing_status += f"\n✅ Excel export created at {excel_file}"
            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}")
                st.session_state.processing_status += f"\n❌ Excel export error: {str(e)}"
        except Exception as e:
            logger.error(f"Error saving parsed CVs: {str(e)}")
                
        return parsed_cvs

def display_cv_info(cv_id, cv_data):
    """Display parsed CV information with enhanced visualization"""
    personal_info = cv_data.get("personal_info", {})
    
    st.subheader(personal_info.get("name", "Unnamed Candidate"))
    
    # Contact and Personal Information section
    with st.expander("📋 Contact & Personal Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Email:**")
            st.write(personal_info.get("email") or "Not provided")
            
            st.write("**Phone:**")
            st.write(personal_info.get("phone") or "Not provided")
            
        with col2:
            st.write("**Location:**")
            st.write(personal_info.get("location") or "Not provided")
            
            st.write("**LinkedIn:**")
            if personal_info.get("linkedin"):
                st.write(f"[Profile]({personal_info.get('linkedin')})")
            else:
                st.write("Not provided")
                
        with col3:
            if "languages" in cv_data and cv_data["languages"]:
                st.write("**Languages:**")
                for lang in cv_data["languages"]:
                    if isinstance(lang, dict):
                        st.write(f"- {lang.get('language', '')}: {lang.get('proficiency', '')}")
                    else:
                        st.write(f"- {lang}")
            
            if personal_info.get("summary"):
                st.write("**Summary:**")
                st.write(personal_info.get("summary"))
    
    # Education section
    with st.expander("🎓 Education", expanded=True):
        education = cv_data.get("education", [])
        if education:
            for edu in education:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    degree = edu.get("degree", "Degree not specified")
                    institution = edu.get("institution", "Institution not specified")
                    
                    st.markdown(f"**{degree}**")
                    st.markdown(f"*{institution}*")
                    
                    if edu.get("field"):
                        st.write(f"Field: {edu.get('field')}")
                        
                    if edu.get("gpa"):
                        st.write(f"GPA: {edu.get('gpa')}")
                        
                with col2:
                    if edu.get("date_range"):
                        st.write(f"**Period:** {edu.get('date_range')}")
                        
                    if edu.get("location"):
                        st.write(f"**Location:** {edu.get('location')}")
                
                if edu.get("achievements"):
                    st.write("**Achievements:**")
                    achievements = edu.get("achievements")
                    if isinstance(achievements, list):
                        for achievement in achievements:
                            st.write(f"- {achievement}")
                    else:
                        st.write(achievements)
                
                st.markdown("---")
        else:
            st.info("No education information provided")
            
    # Experience section
    with st.expander("💼 Work Experience", expanded=True):
        experience = cv_data.get("experience", [])
        if experience:
            for exp in experience:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    job_title = exp.get("job_title", "Position not specified")
                    company = exp.get("company", "Company not specified")
                    
                    st.markdown(f"**{job_title}**")
                    st.markdown(f"*{company}*")
                    
                with col2:
                    if exp.get("date_range"):
                        st.write(f"**Period:** {exp.get('date_range')}")
                        
                    if exp.get("location"):
                        st.write(f"**Location:** {exp.get('location')}")
                
                if "responsibilities" in exp:
                    st.write("**Responsibilities:**")
                    responsibilities = exp.get("responsibilities")
                    if isinstance(responsibilities, list):
                        for resp in responsibilities:
                            st.write(f"- {resp}")
                    else:
                        st.write(responsibilities)
                        
                if "skills_used" in exp:
                    st.write("**Skills Used:**")
                    skills = exp.get("skills_used")
                    if isinstance(skills, list):
                        st.write(", ".join(skills))
                    else:
                        st.write(skills)
                
                st.markdown("---")
        else:
            st.info("No experience information provided")
            
    # Skills section
    with st.expander("🔧 Skills", expanded=True):
        skills = cv_data.get("skills", [])
        if skills:
            # Attempt to categorize skills
            tech_skills = []
            soft_skills = []
            other_skills = []
            
            # Simple heuristic to categorize skills
            tech_keywords = {'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 
                            'angular', 'node', 'cloud', 'aws', 'azure', 'ai', 'ml', 'data', 
                            'analysis', 'programming', 'software', 'development', 'testing'}
            
            soft_keywords = {'communication', 'leadership', 'teamwork', 'management', 'problem-solving',
                           'critical', 'thinking', 'creativity', 'presentation', 'collaboration', 
                           'negotiation', 'adaptation', 'flexibility'}
            
            for skill in skills:
                skill_lower = skill.lower()
                if any(keyword in skill_lower for keyword in tech_keywords):
                    tech_skills.append(skill)
                elif any(keyword in skill_lower for keyword in soft_keywords):
                    soft_skills.append(skill)
                else:
                    other_skills.append(skill)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if tech_skills:
                    st.markdown("**Technical Skills:**")
                    chunks = [tech_skills[i:i+3] for i in range(0, len(tech_skills), 3)]
                    for chunk in chunks:
                        cols = st.columns(3)
                        for i, skill in enumerate(chunk):
                            cols[i].write(f"- {skill}")
            
            with col2:
                if soft_skills:
                    st.markdown("**Soft Skills:**")
                    chunks = [soft_skills[i:i+3] for i in range(0, len(soft_skills), 3)]
                    for chunk in chunks:
                        cols = st.columns(3)
                        for i, skill in enumerate(chunk):
                            cols[i].write(f"- {skill}")
            
            if other_skills:
                st.markdown("**Other Skills:**")
                chunks = [other_skills[i:i+3] for i in range(0, len(other_skills), 3)]
                for chunk in chunks:
                    cols = st.columns(3)
                    for i, skill in enumerate(chunk):
                        cols[i].write(f"- {skill}")
        else:
            st.info("No skills information provided")
            
    # Projects section
    with st.expander("🚀 Projects", expanded=True):
        projects = cv_data.get("projects", [])
        if projects:
            for project in projects:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    project_name = project.get("name", "Project not specified")
                    st.markdown(f"**{project_name}**")
                    
                    if project.get("description"):
                        st.write(project.get("description"))
                        
                    if project.get("technologies"):
                        st.write("**Technologies:**")
                        techs = project.get("technologies")
                        if isinstance(techs, list):
                            st.write(", ".join(techs))
                        else:
                            st.write(techs)
                            
                with col2:
                    if project.get("date_range"):
                        st.write(f"**Period:** {project.get('date_range')}")
                        
                    if project.get("url"):
                        st.write(f"**URL:** [{project.get('url')}]({project.get('url')})")
                
                st.markdown("---")
        else:
            st.info("No projects information provided")
            
    # Certifications section
    with st.expander("🏆 Certifications", expanded=True):
        certifications = cv_data.get("certifications", [])
        if certifications:
            for cert in certifications:
                if isinstance(cert, dict):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        cert_name = cert.get("name", "Certification not specified")
                        st.markdown(f"**{cert_name}**")
                        
                        if cert.get("issuer"):
                            st.write(f"Issued by: {cert.get('issuer')}")
                            
                    with col2:
                        if cert.get("date"):
                            st.write(f"**Date:** {cert.get('date')}")
                            
                        if cert.get("id"):
                            st.write(f"**ID:** {cert.get('id')}")
                else:
                    st.write(f"- {cert}")
            
        else:
            st.info("No certifications information provided")
            
    # Publications section
    if "publications" in cv_data and cv_data["publications"]:
        with st.expander("📚 Publications", expanded=True):
            publications = cv_data["publications"]
            for pub in publications:
                if isinstance(pub, dict):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        pub_title = pub.get("title", "Publication not specified")
                        st.markdown(f"**{pub_title}**")
                        
                        if pub.get("authors"):
                            st.write(f"Authors: {pub.get('authors')}")
                            
                        if pub.get("publisher"):
                            st.write(f"Published in: {pub.get('publisher')}")
                            
                    with col2:
                        if pub.get("date"):
                            st.write(f"**Date:** {pub.get('date')}")
                            
                        if pub.get("url"):
                            st.write(f"**URL:** [{pub.get('url')}]({pub.get('url')})")
                else:
                    st.write(f"- {pub}")
    
    # Document info section
    with st.expander("📄 Document Information", expanded=False):
        doc_info = cv_data.get("document_info", {})
        
        if doc_info:
            st.write(f"**Filename:** {doc_info.get('filename', 'Unknown')}")
            st.write(f"**File Size:** {doc_info.get('file_size', 0) / 1024:.1f} KB")
            st.write(f"**File Type:** {doc_info.get('extension', 'Unknown')}")
            st.write(f"**Scanned Document:** {doc_info.get('is_scanned', False)}")
            st.write(f"**OCR Applied:** {doc_info.get('ocr_applied', False)}")
            st.write(f"**Text Quality:** {doc_info.get('text_quality', 'Unknown')}")

def chat_interface():
    """Chat interface for querying CV information with enhanced prompts"""
    # Display chat history
    for i, (query, response) in enumerate(st.session_state.chat_history):
        message(query, is_user=True, key=f"user_{i}")
        message(response, key=f"assistant_{i}")
        
    # Chat input
    query = st.chat_input("Ask a question about the CVs")
    
    if query:
        # Process the query
        with st.spinner("Thinking..."):
            response, context_id = st.session_state.chatbot.process_query(
                query,
                context_id=st.session_state.context_id,
                job_description=st.session_state.job_description
            )
            
            # Update context ID
            st.session_state.context_id = context_id
            
            # Add to chat history
            st.session_state.chat_history.append((query, response))
            
            # Rerun to update the display
            st.rerun()

def export_data_section():
    """Section for exporting CV data to different formats"""
    st.subheader("Export CV Data")
    
    if not st.session_state.parsed_cvs:
        st.warning("No CVs have been processed yet. Please upload and process CVs first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Select export format:",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
        )
        
        # Select which CVs to export
        cv_options = list(st.session_state.parsed_cvs.keys())
        selected_cvs = st.multiselect(
            "Select CVs to export (leave empty to export all):",
            cv_options
        )
        
        # If none selected, use all
        if not selected_cvs:
            selected_cvs = cv_options
        
        # Store selected CVs in session state
        st.session_state.selected_cvs = selected_cvs
    
    with col2:
        export_filename = st.text_input("Export filename (without extension):", "cv_data_export")
        
        if st.button("Export Data"):
            with st.spinner("Exporting data..."):
                # Filter CVs based on selection
                selected_cv_data = {
                    cv_id: st.session_state.parsed_cvs[cv_id] 
                    for cv_id in selected_cvs
                }
                
                # Create export directory
                export_dir = Path("data/exports")
                export_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    if export_format == "Excel (.xlsx)":
                        output_file = export_dir / f"{export_filename}.xlsx"
                        st.session_state.data_exporter.export_to_excel(selected_cv_data, output_file)
                        st.session_state.export_status = f"✅ Excel file exported to {output_file}"
                    
                    elif export_format == "CSV (.csv)":
                        csv_files = st.session_state.data_exporter.export_to_csv(
                            selected_cv_data, 
                            export_dir,
                            file_prefix=f"{export_filename}_"
                        )
                        st.session_state.export_status = f"✅ CSV files exported to {export_dir}"
                    
                    elif export_format == "JSON (.json)":
                        output_file = export_dir / f"{export_filename}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(selected_cv_data, f, ensure_ascii=False, indent=2)
                        st.session_state.export_status = f"✅ JSON file exported to {output_file}"
                
                except Exception as e:
                    error_msg = f"Error exporting data: {str(e)}"
                    logger.error(error_msg)
                    st.session_state.export_status = f"❌ {error_msg}"
    
    # Display export status
    if st.session_state.export_status:
        st.info(st.session_state.export_status)
        
        # Offer download button if export was successful
        if "✅" in st.session_state.export_status:
            if export_format == "Excel (.xlsx)":
                output_file = export_dir / f"{export_filename}.xlsx"
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download Excel File",
                        data=file,
                        file_name=f"{export_filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            elif export_format == "JSON (.json)":
                output_file = export_dir / f"{export_filename}.json"
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download JSON File",
                        data=file,
                        file_name=f"{export_filename}.json",
                        mime="application/json"
                    )

def main():
    """Main application function with enhanced UI and functionality"""
    # Initialize session state
    init_session_state()
    
    # Set page title and icon
    st.title("📄 Advanced CV Analysis System")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        
        tab_options = ["Upload & Process", "View CVs", "Chat with CVs", "Export Data", "Settings"]
        tab_selection = st.radio("Select a tab:", tab_options, index=tab_options.index(st.session_state.tab_selection))
        st.session_state.tab_selection = tab_selection
        
        # Display job description input in sidebar
        st.title("Job Description")
        job_description = st.text_area(
            "Enter job description for matching:",
            value=st.session_state.job_description,
            height=300
        )
        st.session_state.job_description = job_description
        
        # Reset conversation button
        if st.session_state.chatbot is not None and st.button("Reset Conversation"):
            st.session_state.chat_history = []
            st.session_state.context_id = None
            st.rerun()
        
        # Display usage stats if available
        if st.session_state.chatbot is not None and hasattr(st.session_state.chatbot.llm_service, 'get_usage_stats'):
            with st.expander("LLM Usage Stats", expanded=False):
                try:
                    stats = st.session_state.chatbot.llm_service.get_usage_stats(days=7)
                    st.write(f"Total Requests: {stats['total_requests']}")
                    st.write(f"Total Tokens: {stats['total_tokens_in'] + stats['total_tokens_out']}")
                    st.write(f"Error Rate: {stats['error_rate']:.2%}")
                except Exception as e:
                    st.write(f"Error getting stats: {str(e)}")
    
    # Main content
    if tab_selection == "Upload & Process":
        st.header("Upload & Process CVs")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload CV documents (PDF, DOCX, JPG, JPEG, PNG):",
            type=["pdf", "docx", "doc", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_llm = st.checkbox("Use LLM for enhanced parsing", value=True)
            use_ocr = st.checkbox("Use OCR for scanned documents", value=True)
            
        with col2:
            ocr_language = st.selectbox(
                "OCR Language:",
                ["eng", "fra", "deu", "spa", "ita", "por", "nld", "pol", "rus", "jpn", "chi_sim", "chi_tra", "kor", "ara"],
                index=0
            )
            
            job_position = st.text_input("Job position for matching (optional):", "")
            
        if uploaded_files and st.button("Process CVs"):
            # Process uploaded files
            processed_documents = process_uploaded_files(
                uploaded_files, 
                use_ocr=use_ocr,
                ocr_language=ocr_language
            )
            
            if processed_documents:
                # Parse the processed documents
                parsed_cvs = parse_processed_documents(
                    processed_documents, 
                    use_llm=use_llm,
                    job_position=job_position if job_position else None
                )
                
                if parsed_cvs:
                    st.success(f"Successfully processed and parsed {len(parsed_cvs)} CVs!")
                    st.session_state.tab_selection = "View CVs"
                    st.rerun()
                    
        # Display processing status
        if st.session_state.processing_status:
            with st.expander("Processing Details", expanded=True):
                st.text(st.session_state.processing_status)
                    
        # Display current processed CVs
        if st.session_state.processed_cvs:
            st.subheader("Processed CV Documents")
            st.write(f"Total: {len(st.session_state.processed_cvs)} documents")
            
            for filename, doc_info in st.session_state.processed_cvs.items():
                with st.expander(filename):
                    st.write(f"File size: {doc_info.get('file_size', 0) / 1024:.1f} KB")
                    st.write(f"Scanned PDF: {doc_info.get('is_scanned', False)}")
                    st.write(f"OCR Applied: {doc_info.get('ocr_applied', False)}")
                    st.write(f"Text Quality: {doc_info.get('text_quality', 'unknown')}")
                    st.write(f"Text length: {len(doc_info.get('raw_text', ''))}")
                    
                    if st.button(f"View raw text for {filename}", key=f"raw_{filename}"):
                        with st.expander("Raw Text", expanded=True):
                            st.text_area("", doc_info.get('raw_text', ''), height=200)
    
    elif tab_selection == "View CVs":
        st.header("View Parsed CVs")
        
        if not st.session_state.parsed_cvs:
            st.warning("No CVs have been processed yet. Please upload and process CVs first.")
            return
            
        # Display CV selection
        cv_ids = list(st.session_state.parsed_cvs.keys())
        selected_cv = st.selectbox("Select a CV to view:", cv_ids)
        
        if selected_cv:
            cv_data = st.session_state.parsed_cvs[selected_cv]
            display_cv_info(selected_cv, cv_data)
            
    elif tab_selection == "Chat with CVs":
        st.header("Chat with CV Data")
        
        if not st.session_state.parsed_cvs:
            st.warning("No CVs have been processed yet. Please upload and process CVs first.")
            return
            
        # Initialize LLM service and chatbot if needed
        if st.session_state.chatbot is None:
            llm_service = get_llm_service()
            st.session_state.chatbot = CVChatbot(llm_service, st.session_state.parsed_cvs)
            
        # Display chat interface
        chat_interface()
        
        # Display example queries
        with st.expander("Example Queries"):
            st.markdown("""
            Try asking questions like:
            - Find candidates with skills in Python and data analysis
            - Who has experience in machine learning?
            - Compare education levels of all candidates
            - Which candidate has the most years of experience?
            - Who is the best match for the job description I provided?
            - What certifications do the candidates have?
            - Rank the candidates based on their experience with AWS
            - Which candidate has the most diverse skill set?
            - Find candidates who worked at tech companies
            - Who has a PhD or Master's degree?
            """)
            
    elif tab_selection == "Export Data":
        st.header("Export CV Data")
        
        if not st.session_state.parsed_cvs:
            st.warning("No CVs have been processed yet. Please upload and process CVs first.")
            return
            
        # Display export interface
        export_data_section()
        
    elif tab_selection == "Settings":
        st.header("Settings")
        
        # Theme selection
        theme = st.radio(
            "Select theme:",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1
        )
        
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.experimental_rerun()
        
        # OCR settings
        st.subheader("OCR Settings")
        tesseract_path = st.text_input(
            "Tesseract Path (optional):",
            value=os.getenv("TESSERACT_PATH", "")
        )
        
        if tesseract_path:
            os.environ["TESSERACT_PATH"] = tesseract_path
            st.info(f"Tesseract path set to: {tesseract_path}")
        
        # LLM settings
        st.subheader("LLM Settings")
        
        provider = st.selectbox(
            "LLM Provider:",
            ["groq", "anthropic", "openai"],
            index=0
        )
        
        model_options = {
            "groq": ["llama-3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "openai": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        }
        
        model = st.selectbox(
            "Model:",
            model_options.get(provider, [])
        )
        
        api_key = st.text_input(
            "API Key:",
            type="password"
        )
        
        if st.button("Save LLM Settings"):
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_MODEL"] = model
            
            if api_key:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
                
            st.success("LLM settings saved! Please restart the application for changes to take effect.")

if __name__ == "__main__":
    main()