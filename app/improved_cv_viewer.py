import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def display_cv_info(cv_data: Dict[str, Any], job_position: str = None):
    """
    Display parsed CV information in a structured, visually appealing format.
    
    Args:
        cv_data: Structured CV data
        job_position: Optional job position for context
    """
    # Extract main sections
    personal_info = cv_data.get("personal_information", {})
    education = cv_data.get("education", {})
    work_exp = cv_data.get("work_experience", {})
    skills = cv_data.get("skills", {})
    certifications = cv_data.get("certifications", {})
    rating = cv_data.get("rating", {})
    employment_status = cv_data.get("employment_status", {}).get("current_status")
    
    # Header with personal information
    st.header(personal_info.get("name", "Unnamed Candidate"))
    
    # Candidate rating if available
    if rating and rating.get("score"):
        score = float(rating.get("score", 0))
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Suitability Score", f"{score}/10")
        with col2:
            # Create a gauge chart for the score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Match for {job_position or 'Position'}"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": get_score_color(score)},
                    "steps": [
                        {"range": [0, 4], "color": "lightgray"},
                        {"range": [4, 7], "color": "gray"},
                        {"range": [7, 10], "color": "lightgray"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 7
                    }
                }
            ))
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Contact information
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Contact Information")
            if personal_info.get("email"):
                st.write(f"📧 **Email:** {personal_info.get('email')}")
            if personal_info.get("phone"):
                st.write(f"📱 **Phone:** {personal_info.get('phone')}")
            if personal_info.get("linkedin"):
                st.write(f"🔗 **LinkedIn:** {personal_info.get('linkedin')}")
        
        with col2:
            st.subheader("Location & Background")
            if personal_info.get("current_residence"):
                st.write(f"📍 **Current Location:** {personal_info.get('current_residence')}")
            if personal_info.get("nationality"):
                st.write(f"🌍 **Nationality:** {personal_info.get('nationality')}")
            if employment_status:
                st.write(f"💼 **Employment Status:** {employment_status}")
        
        with col3:
            st.subheader("Languages")
            if personal_info.get("languages") and len(personal_info.get("languages", [])) > 0:
                for language in personal_info.get("languages"):
                    st.write(f"🗣️ {language}")
            else:
                st.write("No language information provided")
    
    st.markdown("---")
    
    # Education section with tabs for different degrees
    st.header("🎓 Education")
    
    # Check if any education info exists
    has_bachelor = education.get("bachelor", {}).get("university")
    has_masters = education.get("masters", {}).get("university")
    has_phd = education.get("phd", {}).get("university")
    
    if has_bachelor or has_masters or has_phd:
        tabs = []
        if has_phd:
            tabs.append("PhD")
        if has_masters:
            tabs.append("Master's")
        if has_bachelor:
            tabs.append("Bachelor's")
            
        if tabs:
            selected_tab = st.radio("Select Degree", tabs, horizontal=True)
            
            if selected_tab == "Bachelor's" and has_bachelor:
                degree_info = education.get("bachelor", {})
                display_degree_info(degree_info, "Bachelor's")
                
            elif selected_tab == "Master's" and has_masters:
                degree_info = education.get("masters", {})
                display_degree_info(degree_info, "Master's")
                
            elif selected_tab == "PhD" and has_phd:
                degree_info = education.get("phd", {})
                display_degree_info(degree_info, "PhD")
    else:
        st.info("No detailed education information available")
    
    st.markdown("---")
    
    # Work Experience Section
    st.header("💼 Work Experience")
    
    years_exp = work_exp.get("years_of_experience")
    companies = work_exp.get("companies", [])
    responsibilities = work_exp.get("top_responsibilities", [])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if years_exp:
            st.metric("Years of Experience", years_exp)
        else:
            st.write("Years of experience not specified")
            
        if companies:
            st.subheader("Companies")
            for company in companies:
                st.write(f"- {company}")
        else:
            st.write("No company information available")
    
    with col2:
        if responsibilities:
            st.subheader("Key Responsibilities & Projects")
            for resp in responsibilities:
                st.write(f"• {resp}")
        else:
            st.write("No detailed responsibilities information available")
    
    st.markdown("---")
    
    # Skills Section
    st.header("🔧 Skills")
    
    # Create tabs for technical and soft skills
    skill_tabs = st.tabs(["Technical Skills", "Soft Skills"])
    
    with skill_tabs[0]:  # Technical Skills
        tech_skills = skills.get("all_technical_skills", [])
        top_tech_skills = skills.get("top_technical_skills", [])
        
        if tech_skills:
            # Create a visualization for technical skills
            df = pd.DataFrame({
                'Skill': tech_skills,
                'Value': [1 if skill in top_tech_skills else 0.5 for skill in tech_skills],
                'Category': ['Top Skill' if skill in top_tech_skills else 'Other Skill' for skill in tech_skills]
            })
            
            fig = px.bar(
                df, 
                x='Skill', 
                y='Value', 
                color='Category',
                color_discrete_map={'Top Skill': '#1f77b4', 'Other Skill': '#aec7e8'},
                title='Technical Skills'
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                yaxis_showticklabels=False,
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No technical skills information available")
    
    with skill_tabs[1]:  # Soft Skills
        soft_skills = skills.get("all_soft_skills", [])
        top_soft_skills = skills.get("top_soft_skills", [])
        
        if soft_skills:
            # Create a visualization for soft skills
            df = pd.DataFrame({
                'Skill': soft_skills,
                'Value': [1 if skill in top_soft_skills else 0.5 for skill in soft_skills],
                'Category': ['Top Skill' if skill in top_soft_skills else 'Other Skill' for skill in soft_skills]
            })
            
            fig = px.bar(
                df, 
                x='Skill', 
                y='Value', 
                color='Category',
                color_discrete_map={'Top Skill': '#2ca02c', 'Other Skill': '#98df8a'},
                title='Soft Skills'
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                yaxis_showticklabels=False,
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No soft skills information available")
    
    st.markdown("---")
    
    # Certifications Section
    st.header("🏆 Certifications & Courses")
    cert_list = certifications.get("top_certifications", [])
    
    if cert_list:
        for cert in cert_list:
            st.write(f"✓ {cert}")
    else:
        st.info("No certification information available")
    
    # If there's rating reasoning, display it
    if rating and rating.get("reasoning"):
        st.markdown("---")
        st.header("📊 Assessment Summary")
        st.write(rating.get("reasoning"))


def display_degree_info(degree_info, degree_type):
    """Display information about a specific degree."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{degree_type} Degree")
        st.write(f"**University:** {degree_info.get('university', 'Not specified')}")
        st.write(f"**Major:** {degree_info.get('major', 'Not specified')}")
    
    with col2:
        st.write(f"**Graduation Date:** {degree_info.get('graduation_date', 'Not specified')}")
        if degree_info.get('gpa'):
            st.write(f"**GPA:** {degree_info.get('gpa')}")


def get_score_color(score):
    """Return color based on score."""
    if score >= 8:
        return "green"
    elif score >= 6:
        return "orange"
    else:
        return "red"