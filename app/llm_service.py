import os
import time
import json
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import sqlite3
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLMService")

# Load environment variables
load_dotenv()

class LLMRateLimiter:
    """
    Rate limiter for LLM API calls with adaptive backoff
    """
    
    def __init__(self, requests_per_minute: int = 30, 
                 max_retries: int = 5, 
                 db_path: str = "rate_limiter.db"):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            max_retries: Maximum number of retries on rate limit errors
            db_path: Path to the SQLite database for tracking requests
        """
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.db_path = db_path
        
        # Set up database for tracking requests
        self._setup_database()
        
        # Last request timestamps
        self.last_request_time = 0
        
        # Jitter for retry
        self.jitter_factor = 0.1
        
        # Tracking rate limit errors
        self.consecutive_rate_limit_errors = 0
    
    def _setup_database(self):
        """Set up the database for tracking request history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for tracking requests
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS request_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            provider TEXT,
            model TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            success BOOLEAN,
            error_type TEXT,
            retry_count INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_request(self, provider: str, model: str, tokens_in: int, 
                      tokens_out: int, success: bool = True, 
                      error_type: Optional[str] = None,
                      retry_count: int = 0):
        """Record a request in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO request_history 
        (timestamp, provider, model, tokens_in, tokens_out, success, error_type, retry_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), provider, model, tokens_in, tokens_out, 
              success, error_type, retry_count))
        
        conn.commit()
        conn.close()
    
    def get_request_count_last_minute(self) -> int:
        """Get the number of requests in the last minute"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        minute_ago = time.time() - 60
        cursor.execute('''
        SELECT COUNT(*) FROM request_history
        WHERE timestamp > ?
        ''', (minute_ago,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def should_rate_limit(self) -> bool:
        """Check if we should rate limit the request"""
        # Check how many requests we've made in the last minute
        recent_requests = self.get_request_count_last_minute()
        
        # If we're over the limit, we should rate limit
        return recent_requests >= self.requests_per_minute
    
    def wait_if_needed(self):
        """Wait if rate limiting is needed"""
        # Check if we need to rate limit
        if self.should_rate_limit():
            # Calculate wait time with jitter
            wait_time = self.request_interval * (1 + random.uniform(0, self.jitter_factor))
            logger.info(f"Rate limiting - waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Ensure minimum time between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def get_retry_delay(self, retry_count: int) -> float:
        """
        Get the retry delay with exponential backoff and jitter.
        
        Args:
            retry_count: Current retry count
            
        Returns:
            float: Delay in seconds
        """
        # Base delay with exponential backoff
        delay = min(60, 2 ** retry_count)
        
        # Add jitter (±10%)
        jitter = delay * random.uniform(-self.jitter_factor, self.jitter_factor)
        
        # If we have consecutive rate limit errors, increase delay
        if self.consecutive_rate_limit_errors > 0:
            delay *= (1 + (0.5 * self.consecutive_rate_limit_errors))
        
        return max(1, delay + jitter)
    
    def handle_rate_limit_error(self):
        """Handle a rate limit error"""
        self.consecutive_rate_limit_errors += 1
    
    def handle_success(self):
        """Handle a successful request"""
        # Reset consecutive rate limit errors
        self.consecutive_rate_limit_errors = 0


# Set up caching to SQLite
cache_path = Path("llm_cache.db")
cache_path.parent.mkdir(exist_ok=True)
set_llm_cache(SQLiteCache(database_path=str(cache_path)))


class LLMService:
    """
    Enhanced service class for interacting with LLM APIs with robust rate limiting,
    error handling, caching, and fallback mechanisms.
    """
    
    def __init__(
        self, 
        provider: str = "groq", 
        model_name: str = "llama-3-8b-8192",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        requests_per_minute: int = 30,
        fallback_providers: Optional[List[Dict[str, str]]] = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            provider: Primary LLM provider ("groq")
            model_name: Model name to use
            api_key: API key (defaults to environment variables)
            max_retries: Maximum number of retries on API errors
            requests_per_minute: Maximum requests per minute (for rate limiting)
            fallback_providers: List of fallback providers and models to use if primary fails
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_retries = max_retries
        
        # Get API keys
        if api_key:
            self.api_key = api_key
        elif self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        if not self.api_key:
            raise ValueError(f"API key for {provider} not found")
            
        # Initialize rate limiter
        self.rate_limiter = LLMRateLimiter(
            requests_per_minute=requests_per_minute,
            max_retries=max_retries
        )
        
        # Set up fallback providers
        self.fallback_providers = fallback_providers or []
        self.current_provider_index = 0
        
        # Initialize the LLM client
        self._initialize_client()
        
        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default for newer models
    
    def _initialize_client(self):
        """Initialize the LLM client based on the provider"""
        # Get current provider info
        if self.current_provider_index == 0:
            # Primary provider
            provider = self.provider
            model_name = self.model_name
            api_key = self.api_key
        else:
            # Fallback provider
            fallback = self.fallback_providers[self.current_provider_index - 1]
            provider = fallback["provider"]
            model_name = fallback["model_name"]
            api_key = fallback.get("api_key") or os.getenv(f"{provider.upper()}_API_KEY")
            
            if not api_key:
                raise ValueError(f"API key for fallback provider {provider} not found")
                
        logger.info(f"Initializing LLM client for provider: {provider}, model: {model_name}")
        
        if provider == "groq":
            self.client = ChatGroq(
                api_key=api_key,
                model_name=model_name,
                temperature=0.3,
                streaming=False
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def switch_to_fallback(self):
        """
        Switch to a fallback provider if available.
        
        Returns:
            bool: True if switched to fallback, False if no more fallbacks
        """
        self.current_provider_index += 1
        
        if self.current_provider_index <= len(self.fallback_providers):
            logger.warning(f"Switching to fallback provider #{self.current_provider_index}")
            self._initialize_client()
            return True
        else:
            # Reset to primary provider for next attempt
            self.current_provider_index = 0
            self._initialize_client()
            return False
            
    def chat(self, messages: List[BaseMessage], timeout: Optional[int] = None) -> Any:
        """
        Send a chat request to the LLM with robust error handling and rate limiting.
        
        Args:
            messages: List of messages to send
            timeout: Optional timeout in seconds
            
        Returns:
            LLM response
        """
        retries = 0
        tokens_in = self.count_tokens("\n".join([m.content for m in messages]))
        
        # Handle rate limiting
        self.rate_limiter.wait_if_needed()
        
        while retries <= self.max_retries:
            try:
                start_time = time.time()
                response = self.client.invoke(messages, timeout=timeout)
                
                # Record successful request
                tokens_out = self.count_tokens(response.content)
                elapsed = time.time() - start_time
                
                logger.info(f"LLM request successful: provider={self.provider}, model={self.model_name}, "
                          f"tokens_in={tokens_in}, tokens_out={tokens_out}, time={elapsed:.2f}s")
                
                self.rate_limiter.record_request(
                    provider=self.provider,
                    model=self.model_name,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=True,
                    retry_count=retries
                )
                
                self.rate_limiter.handle_success()
                return response
                
            except Exception as e:
                retries += 1
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Record failed request
                self.rate_limiter.record_request(
                    provider=self.provider,
                    model=self.model_name,
                    tokens_in=tokens_in,
                    tokens_out=0,
                    success=False,
                    error_type=error_type,
                    retry_count=retries
                )
                
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    # Handle rate limit error
                    self.rate_limiter.handle_rate_limit_error()
                    
                    if retries <= self.max_retries:
                        delay = self.rate_limiter.get_retry_delay(retries)
                        logger.warning(f"Rate limit exceeded. Waiting {delay:.2f} seconds... "
                                     f"({retries}/{self.max_retries})")
                        time.sleep(delay)
                    else:
                        # Try fallback provider
                        if self.switch_to_fallback():
                            logger.info("Trying fallback provider after rate limit errors")
                            retries = 0  # Reset retries for new provider
                        else:
                            logger.error("Rate limit error with all providers")
                            raise
                            
                elif "context length" in error_msg.lower() or "token limit" in error_msg.lower():
                    logger.error(f"Context length error: {error_msg}")
                    raise ValueError("The input is too long for the model's context window. Please reduce input size.")
                    
                elif retries <= self.max_retries:
                    # Other error, retry with standard delay
                    delay = self.rate_limiter.get_retry_delay(retries // 2)  # Less aggressive backoff
                    logger.warning(f"API error: {error_type} - {error_msg}. "
                                 f"Retrying in {delay:.2f} seconds... ({retries}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    # Try fallback provider
                    if self.switch_to_fallback():
                        logger.info(f"Trying fallback provider after error: {error_type}")
                        retries = 0  # Reset retries for new provider
                    else:
                        logger.error(f"Max retries exceeded with all providers: {error_msg}")
                        raise
                    
        raise RuntimeError(f"Failed to get response after {self.max_retries} retries")
    
    def analyze_cv(self, cv_data: Dict[str, Any], query: str) -> str:
        """
        Analyze a parsed CV with a specific query using enhanced prompting.
        
        Args:
            cv_data: Structured CV data
            query: Query to analyze the CV with
            
        Returns:
            str: LLM analysis response
        """
        system_prompt = """
        You are an expert CV/resume analyst and recruiting assistant with years of experience in HR. 
        You'll be given structured data from a CV/resume and a specific query.
        
        Please analyze the CV data carefully and provide a detailed, accurate and helpful analysis based on the query.
        Focus on relevant skills, experience, education, and achievements that relate to the query.
        
        Be specific in your analysis, citing particular elements from the CV that support your conclusions.
        If the CV lacks information relevant to the query, acknowledge this gap clearly.
        
        Format your response in a clear, organized manner using markdown headings and bullet points where appropriate.
        """
        
        cv_json = json.dumps(cv_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        Here is the structured data from a CV/resume:
        ```json
        {cv_json}
        ```
        
        Query: {query}
        
        Please analyze this CV in relation to the query. Focus on relevant skills, experience, and qualifications.
        If there are clear strengths or gaps related to the query, highlight them.
        """
        
        response = self.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content
    
    def compare_cvs(self, cv_data_list: List[Dict[str, Any]], criteria: str) -> str:
        """
        Compare multiple CVs based on specific criteria with enhanced prompting.
        
        Args:
            cv_data_list: List of structured CV data
            criteria: Criteria for comparison
            
        Returns:
            str: LLM comparison response
        """
        system_prompt = """
        You are an expert CV analyst and recruiting assistant with years of experience in talent acquisition.
        You'll be given structured data from multiple CVs/resumes and criteria for comparison.
        
        Please analyze and compare the CVs thoroughly based on the given criteria.
        Focus on relevant skills, experience, education, and achievements that relate to the criteria.
        
        Provide a detailed comparison highlighting:
        1. How each candidate meets or doesn't meet the criteria
        2. Relative strengths and weaknesses between candidates
        3. A clear ranking of candidates according to the criteria
        
        Format your response in a clear, organized manner using markdown headings, tables, 
        and bullet points where appropriate. Include a summary table at the end with 
        candidate rankings and brief justifications.
        """
        
        # Prepare a summarized version of each CV to avoid token limits
        cv_summaries = []
        for i, cv in enumerate(cv_data_list):
            personal_info = cv.get("personal_info", {})
            name = personal_info.get("name", f"Candidate {i+1}")
            
            # Create summary of key information
            summary = {
                "index": i,
                "name": name,
                "personal_info": {
                    "email": personal_info.get("email", ""),
                    "phone": personal_info.get("phone", ""),
                    "location": personal_info.get("location", ""),
                    "linkedin": personal_info.get("linkedin", "")
                },
                "education": cv.get("education", []),
                "skills": cv.get("skills", []),
                "experience": [
                    {
                        "job_title": exp.get("job_title", ""),
                        "company": exp.get("company", ""),
                        "date_range": exp.get("date_range", "")
                    } for exp in cv.get("experience", [])[:3]  # Limit to top 3 experiences
                ],
                "certifications": cv.get("certifications", [])
            }
            
            cv_summaries.append(summary)
            
        summaries_json = json.dumps(cv_summaries, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        Here is structured data from {len(cv_data_list)} CVs/resumes:
        ```json
        {summaries_json}
        ```
        
        Comparison criteria: {criteria}
        
        Please compare these CVs in relation to the criteria and rank them accordingly.
        For each candidate, highlight key strengths and weaknesses related to the criteria.
        Provide a final ranking with clear justifications for your decisions.
        """
        
        response = self.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content
    
    def match_job_requirements(self, cv_data: Dict[str, Any], job_requirements: str) -> str:
        """
        Match CV against job requirements with enhanced prompting.
        
        Args:
            cv_data: Structured CV data
            job_requirements: Job requirements text
            
        Returns:
            str: LLM matching analysis response
        """
        system_prompt = """
        You are an expert CV analyst and recruiting assistant with years of experience in talent acquisition.
        You'll be given structured data from a CV/resume and job requirements.
        
        Your task is to provide a detailed analysis of how well the candidate matches the job requirements.
        
        Please include:
        1. A percentage match score (0-100%) with clear reasoning for the score
        2. A detailed breakdown of how the candidate meets or doesn't meet each key requirement
        3. Highlight of strengths that make the candidate especially suitable
        4. Identification of gaps or missing qualifications
        5. Recommendations for areas where the candidate might need additional training or experience
        
        Format your response in a clear, organized manner using markdown headings, tables, 
        and bullet points where appropriate. Always include a summary section at the top 
        with the match percentage and key findings.
        """
        
        cv_json = json.dumps(cv_data, ensure_ascii=False, indent=2)
        
        user_prompt = f"""
        Here is the structured data from a CV/resume:
        ```json
        {cv_json}
        ```
        
        Job requirements:
        {job_requirements}
        
        Please analyze how well this candidate matches the job requirements.
        Include a percentage match score, highlight the candidate's strengths, 
        identify gaps, and provide recommendations.
        """
        
        response = self.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content
    
    def extract_structured_cv_data(self, cv_text: str) -> Dict[str, Any]:
        """
        Extract highly structured data from CV text with enhanced prompting.
        
        Args:
            cv_text: Raw CV text
            
        Returns:
            dict: Structured CV data
        """
        system_prompt = """
        You are an expert CV/resume parser AI with extensive experience in HR and recruitment.
        You will be given raw text extracted from a CV or resume document.
        Your task is to extract and structure the information into a standardized JSON format.
        
        Please extract the following information and organize it exactly according to the schema below:
        
        1. Personal Information:
           - name: Full name of the candidate
           - email: Email address
           - phone: Phone number
           - location: Current location/address
           - linkedin: LinkedIn profile URL if present
           
        2. Education:
           List of education entries, each containing:
           - institution: Name of educational institution
           - degree: Degree or qualification obtained
           - field: Field of study
           - date_range: Date range of study (e.g., "2015-2019")
           - gpa: GPA if mentioned
           
        3. Experience:
           List of experience entries, each containing:
           - company: Company name
           - job_title: Job title or position
           - date_range: Date range of employment (e.g., "2019-Present")
           - location: Job location if mentioned
           - responsibilities: List of key responsibilities or achievements
           
        4. Skills:
           List of all skills mentioned, both technical and soft skills
           
        5. Projects:
           List of project entries, each containing:
           - name: Project name
           - description: Brief description
           - technologies: Technologies used
           - date_range: Date range if mentioned
           
        6. Certifications:
           List of certification entries, each containing:
           - name: Certification name
           - issuer: Issuing organization
           - date: Date obtained or validity period
           
        7. Languages:
           List of languages with proficiency levels if mentioned
           
        8. Publications:
           List of publications if mentioned
        
        If any section doesn't have information in the CV, include it as an empty list or object.
        Ensure that dates are consistently formatted as YYYY-MM or YYYY format.
        Don't make up or infer information that's not present in the CV text.
        Return the structured data in valid JSON format.
        """
        
        user_prompt = f"""
        Here is the raw text extracted from a CV/resume document:
        
        ```
        {cv_text[:30000]}  # Limit to avoid context length issues
        ```
        
        Please parse this CV text and extract the information according to the specified schema.
        """
        
        response = self.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Extract JSON from the response
        try:
            # Try to find JSON block
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            
            if json_match:
                parsed_data = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                parsed_data = json.loads(response.content)
                
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing LLM JSON response: {str(e)}")
            # Return a simplified structure with the raw response
            return {
                "parsing_error": True,
                "raw_response": response.content,
                "error": str(e)
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage statistics for the LLM service.
        
        Args:
            days: Number of days to get stats for
            
        Returns:
            dict: Usage statistics
        """
        conn = sqlite3.connect(self.rate_limiter.db_path)
        cursor = conn.cursor()
        
        # Calculate start time
        start_time = time.time() - (days * 24 * 60 * 60)
        
        # Get total requests, tokens, and errors
        cursor.execute('''
        SELECT 
            COUNT(*) as total_requests,
            SUM(tokens_in) as total_tokens_in,
            SUM(tokens_out) as total_tokens_out,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as total_errors,
            AVG(retry_count) as avg_retries
        FROM request_history
        WHERE timestamp > ?
        ''', (start_time,))
        
        result = cursor.fetchone()
        
        # Get requests by provider and model
        cursor.execute('''
        SELECT 
            provider,
            model,
            COUNT(*) as requests,
            SUM(tokens_in) as tokens_in,
            SUM(tokens_out) as tokens_out
        FROM request_history
        WHERE timestamp > ?
        GROUP BY provider, model
        ''', (start_time,))
        
        by_provider = []
        for row in cursor.fetchall():
            by_provider.append({
                "provider": row[0],
                "model": row[1],
                "requests": row[2],
                "tokens_in": row[3],
                "tokens_out": row[4]
            })
        
        # Get error types
        cursor.execute('''
        SELECT 
            error_type,
            COUNT(*) as count
        FROM request_history
        WHERE success = 0 AND timestamp > ?
        GROUP BY error_type
        ''', (start_time,))
        
        error_types = []
        for row in cursor.fetchall():
            error_types.append({
                "error_type": row[0] or "Unknown",
                "count": row[1]
            })
        
        conn.close()
        
        return {
            "period_days": days,
            "total_requests": result[0],
            "total_tokens_in": result[1],
            "total_tokens_out": result[2],
            "total_errors": result[3],
            "error_rate": result[3] / result[0] if result[0] > 0 else 0,
            "average_retries": result[4],
            "by_provider": by_provider,
            "error_types": error_types
        }