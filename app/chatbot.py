import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from uuid import uuid4

from langchain.schema import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CVChatbot")

class CVChatbot:
    """
    Enhanced chatbot for querying and analyzing CV information with improved 
    context management for follow-up questions and advanced structured data handling.
    """
    
    def __init__(self, llm_service, cv_data=None):
        """
        Initialize the CV chatbot.
        
        Args:
            llm_service: LLM service for generating responses
            cv_data: Optional dictionary mapping CV IDs to CV data
        """
        self.llm_service = llm_service
        self.cv_data = cv_data or {}
        self.conversation_contexts = {}
        
        # Intent patterns for common queries with improved patterns
        self.intent_patterns = {
            "find_skills": [
                r"find.*with\s+skill[s]?\s+(?:in\s+)?(.+)",
                r"who\s+(?:has|have)\s+experience\s+(?:in|with)\s+(.+)",
                r"candidates?.*(?:know|knowing|skilled in|proficient in|expertise in)\s+(.+)",
                r"find.*(?:familiar|proficient|experienced)(?:\s+with|\s+in)\s+(.+)",
                r"which candidates?(?:\s+can|\s+know how to)?\s+(?:use|work with|code in|program in|develop in)\s+(.+)"
            ],
            "compare_education": [
                r"compare\s+education",
                r"highest\s+(?:education|degree|qualification)",
                r"education\s+level",
                r"academic\s+background",
                r"who\s+has\s+(?:a\s+)?(?:phd|doctorate|master|mba|bachelor|undergraduate|graduate)",
                r"candidates.*(?:university|college|school|degree|education)",
                r"which\s+candidates?\s+(?:studied|graduated|majored)"
            ],
            "work_experience": [
                r"experience\s+(?:in|at|with)\s+(.+)",
                r"worked\s+(?:for|at|with)\s+(.+)",
                r"(?:years|time)\s+(?:of|in)\s+experience",
                r"who\s+has\s+(?:the\s+)?(?:most|more|longest|extensive)\s+experience",
                r"candidates?.*background\s+(?:in|with)\s+(.+)",
                r"employment\s+history",
                r"job\s+history\s+(?:at|with|in)\s+(.+)"
            ],
            "job_match": [
                r"match.*job\s+requirement",
                r"suitable\s+for.*position",
                r"best\s+candidate\s+for",
                r"fit\s+for\s+the\s+role",
                r"who\s+(?:would|should|could|might)\s+(?:be|make)\s+(?:a\s+)?good",
                r"rank\s+candidates\s+(?:for|based)",
                r"top\s+candidates?\s+for",
                r"assess\s+candidates?\s+(?:against|for|based)",
                r"match.*job\s+description"
            ],
            "certifications": [
                r"certif(?:ication|icate|ied)",
                r"which\s+candidates?\s+(?:have|has|possess|hold|earned|obtained|received)\s+(?:a\s+)?certif",
                r"professional\s+credential",
                r"accredited",
                r"licensed",
                r"who\s+is\s+certified\s+in"
            ],
            "languages": [
                r"(?:spoken|programming|foreign|coding)\s+language",
                r"who\s+(?:speaks|knows|can speak|is fluent in|understands)\s+(.+)",
                r"multilingual",
                r"(?:proficiency|fluency)\s+in\s+(.+)"
            ],
            "projects": [
                r"project",
                r"portfolio",
                r"who\s+(?:has|worked on|built|developed|created|contributed to)\s+(?:a\s+)?project",
                r"personal\s+project",
                r"side\s+project",
                r"project\s+experience",
                r"project\s+involving\s+(.+)"
            ]
        }
    
    def _create_conversation_context(self) -> str:
        """
        Create a new conversation context.
        
        Returns:
            str: Conversation context ID
        """
        context_id = str(uuid4())
        self.conversation_contexts[context_id] = {
            "current_cvs": set(self.cv_data.keys()),
            "last_query": None,
            "last_intent": None,
            "last_entities": [],
            "history": [],
            "focus_candidates": None,  # For tracking specific candidates in focus
            "focus_skills": None,      # For tracking specific skills in focus
            "focus_companies": None,   # For tracking specific companies in focus
            "focus_attribute": None,   # For tracking a specific attribute (education, experience, etc.)
            "job_description": None    # For storing job description context
        }
        return context_id
    
    def _detect_intent(self, query: str) -> Tuple[Optional[str], List[str]]:
        """
        Detect intent and entities from the query with improved accuracy.
        
        Args:
            query: User query
            
        Returns:
            tuple: (intent, entities)
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    entities = [group for group in match.groups() if group]
                    return intent, entities
                    
        return None, []
    
    def _analyze_query_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the query to extract additional context information.
        
        Args:
            query: User query
            context: Current conversation context
            
        Returns:
            dict: Updated context information
        """
        query_lower = query.lower()
        
        # Check for comparisons
        if any(term in query_lower for term in ["compare", "vs", "versus", "better", "best", "strongest", "weakest", "most", "least"]):
            context["is_comparison"] = True
        else:
            context["is_comparison"] = False
        
        # Check for quantitative queries
        if any(term in query_lower for term in ["how many", "count", "number of", "total", "sum"]):
            context["is_quantitative"] = True
        else:
            context["is_quantitative"] = False
        
        # Check for ranking queries
        if any(term in query_lower for term in ["rank", "top", "best", "order", "sort", "list"]):
            context["is_ranking"] = True
        else:
            context["is_ranking"] = False
        
        # Check for specific attribute focus
        attribute_patterns = {
            "education": [r'education', r'degree', r'university', r'college', r'academic', r'studied', r'graduated'],
            "experience": [r'experience', r'job', r'work', r'career', r'employment', r'position'],
            "skills": [r'skill', r'ability', r'proficiency', r'competency', r'capable', r'knowledge'],
            "certifications": [r'certification', r'certificate', r'credential', r'licensed', r'accredited'],
            "languages": [r'language', r'speak', r'fluent', r'bilingual', r'multilingual'],
            "projects": [r'project', r'portfolio', r'built', r'developed', r'created']
        }
        
        for attribute, patterns in attribute_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                context["focus_attribute"] = attribute
                break
        
        return context
    
    def _filter_cvs_by_intent(
        self,
        intent: Optional[str],
        entities: List[str],
        cv_set: Set[str],
        context: Dict[str, Any]
    ) -> Set[str]:
        """
        Filter CVs based on detected intent and entities with improved matching.
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            cv_set: Set of CV IDs to filter
            context: Conversation context
            
        Returns:
            set: Filtered set of CV IDs
        """
        if not intent and not entities and not context.get("focus_attribute"):
            return cv_set
            
        filtered_cvs = set()
        
        if intent == "find_skills":
            # Extract skills from entities
            skill_terms = []
            for entity in entities:
                # Split by common delimiters
                terms = re.split(r',|\sand\s|\sor\s|/|;', entity)
                skill_terms.extend([term.strip().lower() for term in terms if term.strip()])
            
            # Track focused skills for follow-up questions
            context["focus_skills"] = skill_terms
            
            for cv_id in cv_set:
                cv = self.cv_data.get(cv_id, {})
                
                # Get skills from various possible structures
                all_skills = []
                
                # Check traditional structure
                if "skills" in cv and isinstance(cv["skills"], list):
                    all_skills.extend([skill.lower() for skill in cv["skills"]])
                
                # Check enhanced structure
                if "skills" in cv and isinstance(cv["skills"], dict):
                    for skill_type in ["technical_skills", "soft_skills", "top_skills", "all_skills", 
                                      "top_technical_skills", "top_soft_skills", 
                                      "all_technical_skills", "all_soft_skills"]:
                        if skill_type in cv["skills"]:
                            skills_list = cv["skills"][skill_type]
                            if isinstance(skills_list, list):
                                all_skills.extend([skill.lower() for skill in skills_list])
                
                # Check experience section for mentioned skills
                if "experience" in cv:
                    for exp in cv.get("experience", []):
                        # Check for skills used field
                        if "skills_used" in exp:
                            skills_used = exp["skills_used"]
                            if isinstance(skills_used, list):
                                all_skills.extend([skill.lower() for skill in skills_used])
                            elif isinstance(skills_used, str):
                                all_skills.append(skills_used.lower())
                        
                        # Check for skills in responsibilities
                        if "responsibilities" in exp:
                            resp_text = ""
                            if isinstance(exp["responsibilities"], list):
                                resp_text = " ".join(exp["responsibilities"])
                            elif isinstance(exp["responsibilities"], str):
                                resp_text = exp["responsibilities"]
                            
                            # Check if skill terms are mentioned in responsibilities
                            for skill in skill_terms:
                                if skill in resp_text.lower():
                                    all_skills.append(skill)
                
                # Check projects section for mentioned skills
                if "projects" in cv:
                    for project in cv.get("projects", []):
                        # Check for technologies field
                        if "technologies" in project:
                            techs = project["technologies"]
                            if isinstance(techs, list):
                                all_skills.extend([tech.lower() for tech in techs])
                            elif isinstance(techs, str):
                                all_skills.append(techs.lower())
                
                # Remove duplicates
                all_skills = list(set(all_skills))
                
                # Check if any of the skill terms are in the CV skills
                found_match = False
                for term in skill_terms:
                    for skill in all_skills:
                        if term in skill or skill in term:
                            found_match = True
                            break
                    
                    if found_match:
                        break
                
                if found_match:
                    filtered_cvs.add(cv_id)
                    
        elif intent == "compare_education":
            # For education comparison, filter based on education keywords if any
            education_keywords = []
            for entity in entities:
                terms = re.split(r',|\sand\s|\sor\s|/|;', entity)
                education_keywords.extend([term.strip().lower() for term in terms if term.strip()])
            
            # Set focus attribute for follow-up questions
            context["focus_attribute"] = "education"
            
            if education_keywords:
                for cv_id in cv_set:
                    cv = self.cv_data.get(cv_id, {})
                    education = cv.get("education", [])
                    
                    if not education:
                        continue
                    
                    # Check traditional structure
                    if isinstance(education, list):
                        education_text = " ".join([
                            str(edu.get("degree", "")) + " " + 
                            str(edu.get("institution", "")) + " " +
                            str(edu.get("field", ""))
                            for edu in education
                        ]).lower()
                    
                    # Check enhanced structure
                    elif isinstance(education, dict):
                        education_text = ""
                        for level in ["phd", "masters", "bachelor"]:
                            if level in education:
                                edu = education[level]
                                education_text += " ".join([
                                    str(edu.get("university", "")),
                                    str(edu.get("major", "")),
                                    level
                                ])
                    else:
                        education_text = ""
                    
                    if any(keyword in education_text for keyword in education_keywords):
                        filtered_cvs.add(cv_id)
            else:
                # If no specific education keywords, include all CVs with education info
                for cv_id in cv_set:
                    cv = self.cv_data.get(cv_id, {})
                    education = cv.get("education", [])
                    if education:
                        filtered_cvs.add(cv_id)
            
        elif intent == "work_experience":
            # Extract company/industry terms
            company_terms = []
            for entity in entities:
                terms = re.split(r',|\sand\s|\sor\s|/|;', entity)
                company_terms.extend([term.strip().lower() for term in terms if term.strip()])
            
            # Track focused companies for follow-up questions
            context["focus_companies"] = company_terms
            
            # Set focus attribute
            context["focus_attribute"] = "experience"
            
            for cv_id in cv_set:
                cv = self.cv_data.get(cv_id, {})
                
                # Check traditional structure
                if "experience" in cv and isinstance(cv["experience"], list):
                    experience = cv["experience"]
                    
                    for exp in experience:
                        company = exp.get("company", "").lower()
                        job_title = exp.get("job_title", "").lower()
                        responsibilities = exp.get("responsibilities", [])
                        
                        if isinstance(responsibilities, list):
                            resp_text = " ".join(responsibilities).lower()
                        else:
                            resp_text = str(responsibilities).lower()
                        
                        combined_text = company + " " + job_title + " " + resp_text
                        
                        if company_terms:
                            # Check if any company terms match
                            if any(term in combined_text for term in company_terms):
                                filtered_cvs.add(cv_id)
                                break
                        else:
                            # If no specific terms, include all CVs with experience
                            filtered_cvs.add(cv_id)
                            break
                
                # Check enhanced structure
                elif "work_experience" in cv and isinstance(cv["work_experience"], dict):
                    companies = cv["work_experience"].get("companies", [])
                    
                    if company_terms:
                        if any(any(term in company.lower() for company in companies) for term in company_terms):
                            filtered_cvs.add(cv_id)
                    else:
                        # If no specific terms, include all CVs with experience
                        if companies:
                            filtered_cvs.add(cv_id)
                            
        elif intent == "job_match":
            # For job matching, return all CVs for matching against job description
            # Set focus attribute
            context["focus_attribute"] = "job_match"
            return cv_set
            
        elif intent == "certifications":
            # Set focus attribute for follow-up questions
            context["focus_attribute"] = "certifications"
            
            for cv_id in cv_set:
                cv = self.cv_data.get(cv_id, {})
                certifications = cv.get("certifications", [])
                
                if certifications:
                    filtered_cvs.add(cv_id)
                    
        elif intent == "languages":
            # Extract language terms
            language_terms = []
            for entity in entities:
                terms = re.split(r',|\sand\s|\sor\s|/|;', entity)
                language_terms.extend([term.strip().lower() for term in terms if term.strip()])
            
            # Set focus attribute for follow-up questions
            context["focus_attribute"] = "languages"
            
            for cv_id in cv_set:
                cv = self.cv_data.get(cv_id, {})
                
                # Check if languages field exists
                if "languages" in cv:
                    languages = cv["languages"]
                    
                    if isinstance(languages, list):
                        if language_terms:
                            # Check for matches with specific languages
                            found_match = False
                            for lang_item in languages:
                                if isinstance(lang_item, dict):
                                    lang = lang_item.get("language", "").lower()
                                else:
                                    lang = str(lang_item).lower()
                                
                                if any(term in lang for term in language_terms):
                                    found_match = True
                                    break
                            
                            if found_match:
                                filtered_cvs.add(cv_id)
                        else:
                            # If no specific languages requested, include all with language info
                            filtered_cvs.add(cv_id)
                
                # Check personal_info for languages
                elif "personal_info" in cv and "languages" in cv["personal_info"]:
                    languages = cv["personal_info"]["languages"]
                    
                    if isinstance(languages, list):
                        if language_terms:
                            # Check for matches with specific languages
                            if any(any(term in lang.lower() for term in language_terms) for lang in languages):
                                filtered_cvs.add(cv_id)
                        else:
                            # If no specific languages requested, include all with language info
                            filtered_cvs.add(cv_id)
                            
        elif intent == "projects":
            # Set focus attribute for follow-up questions
            context["focus_attribute"] = "projects"
            
            # Extract project-related terms
            project_terms = []
            for entity in entities:
                terms = re.split(r',|\sand\s|\sor\s|/|;', entity)
                project_terms.extend([term.strip().lower() for term in terms if term.strip()])
            
            for cv_id in cv_set:
                cv = self.cv_data.get(cv_id, {})
                projects = cv.get("projects", [])
                
                if isinstance(projects, list):
                    if project_terms:
                        # Check for matches with specific project terms
                        found_match = False
                        for project in projects:
                            if isinstance(project, dict):
                                name = project.get("name", "").lower()
                                desc = project.get("description", "").lower()
                                techs = project.get("technologies", [])
                                
                                if isinstance(techs, list):
                                    techs_text = " ".join(techs).lower()
                                else:
                                    techs_text = str(techs).lower()
                                
                                combined_text = name + " " + desc + " " + techs_text
                                
                                if any(term in combined_text for term in project_terms):
                                    found_match = True
                                    break
                        
                        if found_match:
                            filtered_cvs.add(cv_id)
                    else:
                        # If no specific project terms, include all with project info
                        filtered_cvs.add(cv_id)
        
        return filtered_cvs if filtered_cvs else cv_set
    
    def _get_cvs_for_query(self, query: str, context_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get relevant CVs for the query using the conversation context with enhanced context handling.
        
        Args:
            query: User query
            context_id: Conversation context ID
            
        Returns:
            tuple: (list of relevant CV data, updated context)
        """
        context = self.conversation_contexts.get(context_id)
        if not context:
            return [], {}
            
        # Detect intent and entities
        intent, entities = self._detect_intent(query)
        
        # Analyze query for additional context
        context = self._analyze_query_context(query, context)
        
        # Check for follow-up questions
        is_followup = self._is_followup_question(query, context)
        
        # Update context
        context["last_query"] = query
        context["last_intent"] = intent
        context["last_entities"] = entities
        context["is_followup"] = is_followup
        context["history"].append({
            "query": query, 
            "intent": intent, 
            "entities": entities,
            "is_followup": is_followup
        })
        
        # Initial CV set based on context
        if is_followup and context.get("focus_candidates"):
            # For follow-up questions, use the previously focused candidates
            cv_set = context["focus_candidates"]
        else:
            # Otherwise, use all CVs
            cv_set = context["current_cvs"]
        
        # Filter CVs based on intent and entities
        filtered_cv_ids = self._filter_cvs_by_intent(intent, entities, cv_set, context)
        
        # Update focus candidates for future follow-up questions
        context["focus_candidates"] = filtered_cv_ids
        context["current_cvs"] = filtered_cv_ids
        
        # Get the CV data for the filtered IDs
        filtered_cvs = [self.cv_data[cv_id] for cv_id in filtered_cv_ids]
        
        return filtered_cvs, context
    
    def _is_followup_question(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Determine if the current query is a follow-up question.
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            bool: Whether the query is a follow-up question
        """
        # If no previous query, it's not a follow-up
        if not context.get("last_query"):
            return False
        
        query_lower = query.lower()
        
        # Check for explicit references to previous query
        explicit_followup_markers = [
            "which", "who", "them", "they", "those", "these", "that", "their",
            "of those", "among them", "from those", "between them",
            "how many", "what about", "and what", "what else", "also", "too"
        ]
        
        if any(marker in query_lower for marker in explicit_followup_markers):
            return True
        
        # Check for anaphoric references (pronouns referring to previous entities)
        anaphoric_markers = ["he", "she", "it", "this", "that"]
        if any(marker in query_lower.split() for marker in anaphoric_markers):
            return True
        
        # Check for implied continuations (missing subject)
        words = query_lower.split()
        if len(words) > 0:
            # Check if query starts with a verb
            first_word = words[0]
            common_initial_verbs = ["has", "have", "is", "are", "do", "does", "did", "can", "could", "compare", "rank", "list", "show"]
            if first_word in common_initial_verbs:
                return True
        
        # Check if the query is very short (likely a continuation)
        if len(query_lower.split()) <= 3:
            return True
        
        # Not a follow-up
        return False
    
    def _generate_system_prompt(self, context_id: str, relevant_cvs: List[Dict[str, Any]]) -> str:
        """
        Generate system prompt based on conversation context with improved instructions.
        
        Args:
            context_id: Conversation context ID
            relevant_cvs: List of relevant CV data
            
        Returns:
            str: System prompt
        """
        context = self.conversation_contexts.get(context_id)
        if not context:
            return ""
            
        prompt = """
        You are an expert CV/resume analyst AI assistant specializing in HR and recruitment with years of professional experience.
        You have access to structured data from multiple CVs/resumes and can provide detailed insights, comparisons, and match assessments.
        
        Guidelines:
        1. Answer questions clearly, accurately, and comprehensively based on the CV data provided.
        2. Provide specific examples and details from the CVs to support your analysis.
        3. When comparing candidates, highlight key differentiating factors and provide a clear assessment.
        4. If asked about job fit, analyze both qualifications and potential gaps based on the job description.
        5. If you don't have enough information to answer a question, acknowledge this and explain what would be needed.
        6. Format your responses using markdown when appropriate for readability (headings, bullet points, tables).
        7. Use a professional, balanced tone appropriate for HR and recruitment contexts.
        8. Focus on relevant information only - don't overwhelm with unnecessary details.
        9. Never make assumptions beyond what's in the CV data.
        10. Respect privacy by not sharing full contact details when summarizing CVs.
        
        Important: Each CV entry contains structured information about a candidate, including personal details, education, work experience, skills, and possibly projects/certifications. Use this data to provide informed answers.
        """
        
        # Add focus information if available
        if context.get("focus_attribute"):
            focus_attributes = {
                "education": "education history, degrees, qualifications, institutions, fields of study",
                "experience": "work experience, job roles, responsibilities, companies, industries, years of experience",
                "skills": "technical skills, soft skills, competencies, proficiencies, technologies",
                "certifications": "professional certifications, credentials, licenses, accreditations",
                "languages": "language proficiency, multilingual abilities, spoken/written languages",
                "projects": "projects, portfolios, implementations, developments, creations",
                "job_match": "matching candidate qualifications to job requirements, assessing suitability for roles"
            }
            
            if context["focus_attribute"] in focus_attributes:
                prompt += f"\n\nThe current conversation is focused on {focus_attributes[context['focus_attribute']]}. Pay special attention to this information in your response."
        
        # Add context from conversation history
        if context["history"]:
            prompt += "\n\nConversation history:\n"
            
            # Add last 3 entries for context
            for i, entry in enumerate(context["history"][-3:]):
                prompt += f"- User asked: \"{entry['query']}\"\n"
                
                if entry.get("is_followup"):
                    prompt += f"  (This was a follow-up question)\n"
                
                if entry.get("intent"):
                    prompt += f"  (Intent: {entry['intent']})\n"
                    
        # Add information about the number of CVs
        prompt += f"\n\nYou currently have {len(relevant_cvs)} CVs that match the query criteria."
        
        return prompt
    
    def process_query(self, query: str, context_id: Optional[str] = None, job_description: Optional[str] = None) -> Tuple[str, str]:
        """
        Process a user query and generate a response with enhanced context management.
        
        Args:
            query: User query
            context_id: Optional conversation context ID
            job_description: Optional job description for job matching queries
            
        Returns:
            tuple: (response, context_id)
        """
        # Create new context if not provided
        if not context_id or context_id not in self.conversation_contexts:
            context_id = self._create_conversation_context()
            
        # Store job description in context if provided
        if job_description:
            self.conversation_contexts[context_id]["job_description"] = job_description
            
        # Get relevant CVs for the query
        relevant_cvs, updated_context = self._get_cvs_for_query(query, context_id)
        
        if not relevant_cvs:
            return "I couldn't find any CVs matching your query. Try a different search criteria or check if the CV data contains the information you're looking for.", context_id
            
        # Generate system prompt
        system_prompt = self._generate_system_prompt(context_id, relevant_cvs)
        
        # Prepare user prompt with CV data
        user_prompt = f"Query: {query}\n\n"
        
        if len(relevant_cvs) == 1:
            # Single CV query
            cv = relevant_cvs[0]
            cv_json = json.dumps(cv, ensure_ascii=False, indent=2)
            user_prompt += f"CV data:\n```json\n{cv_json}\n```\n\n"
            
            # Add job description if provided and query involves job matching
            if job_description and (
                updated_context.get("focus_attribute") == "job_match" or 
                updated_context.get("last_intent") == "job_match" or
                "match" in query.lower() or "fit" in query.lower() or "suitable" in query.lower()
            ):
                user_prompt += f"Job description:\n{job_description}\n\n"
                
        else:
            # Multiple CV query
            user_prompt += f"I have {len(relevant_cvs)} CVs that match your query.\n\n"
            
            # For comparison queries, include summarized CV data
            if len(relevant_cvs) <= 8:  # Limit to avoid token limits
                cv_summaries = []
                
                for i, cv in enumerate(relevant_cvs):
                    # Get personal info
                    personal_info = cv.get("personal_info", {})
                    name = personal_info.get("name", f"Candidate {i+1}")
                    
                    # Create appropriate summary based on focus attribute
                    focus_attribute = updated_context.get("focus_attribute")
                    
                    if focus_attribute == "education":
                        # Include detailed education info
                        education = cv.get("education", [])
                        summary = {
                            "name": name,
                            "education": education
                        }
                    elif focus_attribute == "experience":
                        # Include detailed experience info
                        experience = cv.get("experience", [])
                        work_exp = cv.get("work_experience", {})
                        
                        if experience:
                            # Traditional structure
                            summary = {
                                "name": name,
                                "experience": experience
                            }
                        elif work_exp:
                            # Enhanced structure
                            summary = {
                                "name": name,
                                "work_experience": work_exp
                            }
                        else:
                            summary = {
                                "name": name,
                                "experience": []
                            }
                    elif focus_attribute == "skills":
                        # Include detailed skills info
                        skills = cv.get("skills", [])
                        summary = {
                            "name": name,
                            "skills": skills
                        }
                    elif focus_attribute == "certifications":
                        # Include detailed certifications info
                        certifications = cv.get("certifications", [])
                        summary = {
                            "name": name,
                            "certifications": certifications
                        }
                    elif focus_attribute == "languages":
                        # Include detailed language info
                        languages = cv.get("languages", [])
                        summary = {
                            "name": name,
                            "languages": languages
                        }
                    elif focus_attribute == "projects":
                        # Include detailed project info
                        projects = cv.get("projects", [])
                        summary = {
                            "name": name,
                            "projects": projects
                        }
                    elif focus_attribute == "job_match":
                        # Include comprehensive info for job matching
                        summary = {
                            "name": name,
                            "personal_info": {
                                "email": personal_info.get("email", ""),
                                "location": personal_info.get("location", "")
                            },
                            "education": cv.get("education", []),
                            "experience": cv.get("experience", []),
                            "skills": cv.get("skills", []),
                            "certifications": cv.get("certifications", [])
                        }
                    else:
                        # Default summary with key info
                        summary = {
                            "name": name,
                            "education": cv.get("education", []),
                            "skills": cv.get("skills", []),
                            "experience": [
                                {
                                    "job_title": exp.get("job_title", ""),
                                    "company": exp.get("company", ""),
                                    "date_range": exp.get("date_range", "")
                                } for exp in cv.get("experience", [])[:2]  # Limit to top 2 experiences
                            ],
                            "certifications": cv.get("certifications", [])[:3]  # Limit to top 3 certifications
                        }
                    
                    cv_summaries.append(summary)
                    
                summaries_json = json.dumps(cv_summaries, ensure_ascii=False, indent=2)
                user_prompt += f"CV summaries:\n```json\n{summaries_json}\n```\n\n"
                
            # Add job description if provided and query involves job matching
            if job_description and (
                updated_context.get("focus_attribute") == "job_match" or 
                updated_context.get("last_intent") == "job_match" or
                "match" in query.lower() or "fit" in query.lower() or "suitable" in query.lower()
            ):
                user_prompt += f"Job description:\n{job_description}\n\n"
                
        # For follow-up questions, include previous context
        if updated_context.get("is_followup"):
            # Add explicit instruction for handling follow-up
            user_prompt += "Note: This is a follow-up question to the previous query. Please consider the conversation context in your response.\n\n"
            
            # Add specific context based on focus
            if updated_context.get("focus_skills"):
                user_prompt += f"Previous context: The conversation is focused on skills related to: {', '.join(updated_context['focus_skills'])}\n\n"
                
            if updated_context.get("focus_companies"):
                user_prompt += f"Previous context: The conversation is focused on companies/industries related to: {', '.join(updated_context['focus_companies'])}\n\n"
                
            if updated_context.get("focus_attribute"):
                user_prompt += f"Previous context: The conversation is focused on {updated_context['focus_attribute']}.\n\n"
                
        # For comparison queries, add specific instructions
        if updated_context.get("is_comparison"):
            user_prompt += "This query involves comparing candidates. Please provide a structured comparison highlighting key similarities and differences.\n\n"
            
        # For ranking queries, add specific instructions
        if updated_context.get("is_ranking"):
            user_prompt += "This query involves ranking candidates. Please provide a clear ranking with justification for each candidate's position.\n\n"
        
        try:
            # Get response from LLM
            response = self.llm_service.chat([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            return response.content, context_id
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return f"I encountered an error while processing your query. Please try again or rephrase your question. Error details: {str(e)}", context_id
    
    def add_cv(self, cv_id: str, cv_data: Dict[str, Any]):
        """
        Add a CV to the chatbot's data.
        
        Args:
            cv_id: CV identifier
            cv_data: Structured CV data
        """
        self.cv_data[cv_id] = cv_data
        
        # Update all conversation contexts to include the new CV
        for context in self.conversation_contexts.values():
            context["current_cvs"].add(cv_id)
    
    def remove_cv(self, cv_id: str):
        """
        Remove a CV from the chatbot's data.
        
        Args:
            cv_id: CV identifier
        """
        if cv_id in self.cv_data:
            del self.cv_data[cv_id]
            
            # Update all conversation contexts to remove the CV
            for context in self.conversation_contexts.values():
                context["current_cvs"].discard(cv_id)
                if "focus_candidates" in context and cv_id in context["focus_candidates"]:
                    context["focus_candidates"].discard(cv_id)
    
    def reset_conversation(self, context_id: str):
        """
        Reset a conversation context.
        
        Args:
            context_id: Conversation context ID
        """
        if context_id in self.conversation_contexts:
            # Preserve job description if it exists
            job_description = self.conversation_contexts[context_id].get("job_description")
            
            self.conversation_contexts[context_id] = {
                "current_cvs": set(self.cv_data.keys()),
                "last_query": None,
                "last_intent": None,
                "last_entities": [],
                "history": [],
                "focus_candidates": None,
                "focus_skills": None,
                "focus_companies": None,
                "focus_attribute": None,
                "job_description": job_description
            }
    
    def get_suggested_queries(self, context_id: Optional[str] = None) -> List[str]:
        """
        Get a list of suggested queries based on the CV data and conversation context.
        
        Args:
            context_id: Optional conversation context ID
            
        Returns:
            list: List of suggested queries
        """
        # Basic suggestions for new conversations
        basic_suggestions = [
            "Find candidates with Python and Data Science skills",
            "Who has the highest education level?",
            "Compare all candidates' work experience",
            "Which candidate is the best match for a Software Engineer position?",
            "Show me candidates with Master's degrees",
            "Who has experience at top tech companies?",
            "Find candidates who speak multiple languages",
            "Which candidate has the most years of experience?"
        ]
        
        # If no context_id or not in conversation_contexts, return basic suggestions
        if not context_id or context_id not in self.conversation_contexts:
            return basic_suggestions
            
        context = self.conversation_contexts[context_id]
        
        # If focus_attribute is set, provide attribute-specific follow-up suggestions
        if context.get("focus_attribute"):
            attribute = context["focus_attribute"]
            
            if attribute == "education":
                return [
                    "Who has the highest degree?",
                    "Compare their academic backgrounds",
                    "Which universities did they attend?",
                    "Who studied Computer Science?",
                    "Who has the most recent education?"
                ]
            elif attribute == "experience":
                return [
                    "Who has the most years of experience?",
                    "Compare their job responsibilities",
                    "Who has worked at the largest companies?",
                    "Which skills did they use in their jobs?",
                    "Who has leadership experience?"
                ]
            elif attribute == "skills":
                # If focus_skills exists, suggest skill-specific queries
                if context.get("focus_skills"):
                    skills = context["focus_skills"][:2]  # Limit to 2 skills for suggestions
                    suggestions = []
                    
                    for skill in skills:
                        suggestions.extend([
                            f"Who has the most experience with {skill}?",
                            f"Where did they use {skill} in their work?",
                            f"What projects involved {skill}?"
                        ])
                    
                    suggestions.extend([
                        "What other technical skills do they have?",
                        "Compare their skill sets",
                        "Who has the most diverse skill set?"
                    ])
                    
                    return suggestions[:8]  # Limit to 8 suggestions
                else:
                    return [
                        "Which technical skills are most common?",
                        "Who has the most diverse skill set?",
                        "Compare their programming languages",
                        "Who has soft skills like leadership?",
                        "Which rare or specialized skills do they have?"
                    ]
            elif attribute == "certifications":
                return [
                    "Who has the most certifications?",
                    "Compare their professional certifications",
                    "Who has cloud certifications?",
                    "When did they obtain their certifications?",
                    "Which certifications are most relevant for a tech role?"
                ]
            elif attribute == "languages":
                return [
                    "Who speaks the most languages?",
                    "Who is fluent in multiple languages?",
                    "Compare their language proficiencies",
                    "Which languages are most common among them?",
                    "Who can speak both English and Spanish?"
                ]
            elif attribute == "projects":
                return [
                    "Who has the most impressive projects?",
                    "Compare their technical projects",
                    "Which technologies did they use in their projects?",
                    "Who has open source contributions?",
                    "Which projects are most relevant for a tech role?"
                ]
            elif attribute == "job_match":
                return [
                    "Who is the best match for this position?",
                    "What skills are missing from the candidates?",
                    "Rank the candidates by fit for this role",
                    "Who has the most relevant experience?",
                    "Compare the top 3 candidates for this position"
                ]
        
        # If we have focused candidates, provide candidate-specific suggestions
        if context.get("focus_candidates") and len(context["focus_candidates"]) <= 3:
            return [
                "Compare their education backgrounds",
                "Who has more work experience?",
                "Which skills do they share?",
                "Who is more suitable for a tech leadership role?",
                "What are their unique strengths?",
                "Compare their project experiences",
                "Who has more specialized technical skills?"
            ]
            
        # Default to basic suggestions
        return basic_suggestions