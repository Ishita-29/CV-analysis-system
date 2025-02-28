# Advanced CV Analysis System

A comprehensive system for processing, analyzing, and querying CV/resume documents with AI-powered parsing and a chatbot interface.

## Features

- **Document Processing**
  - Support for PDF, DOCX, and image-based resumes (PNG, JPG)
  - OCR for scanned documents using Tesseract
  - Robust text extraction with preprocessing for better quality

- **Structured CV Parsing**
  - Detailed extraction of personal information, education, work experience, and skills
  - GPA, graduation dates, and degree majors extraction
  - Technical vs. soft skills classification
  - Employment status detection

- **LLM-Powered Analysis**
  - Integration with Groq API for advanced language processing
  - Detailed candidate rating and position suitability analysis
  - Error-resistant implementation with graceful fallbacks

- **Interactive Chat Interface**
  - Natural language querying of CV information
  - Context-aware follow-up questions handling
  - Suggested queries for better user experience

- **Modern Streamlit UI**
  - Dashboard with statistics and visualizations
  - Intuitive navigation and user controls
  - Responsive design with custom styling
  - Detailed CV information display with visual elements

## System Architecture

```
cv-analysis-system/
├── app/
│   ├── main.py                  # Main Streamlit application
│   ├── document_processor.py    # Document handling and OCR
│   ├── cv_parser.py             # Structured CV information extraction
│   ├── llm_service.py           # LLM API integration
│   ├── chatbot.py               # Chatbot query processing
│   ├── improved_cv_viewer.py    # Enhanced CV visualization
│   └── utils.py                 # Helper functions
├── data/
│   ├── sample_cvs/              # Sample CV documents
│   └── processed/               # Extracted and structured CV data
├── .env                         # Environment variables (API keys)
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR (for image-based and scanned documents)
- Groq API key or other supported LLM provider

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Ishita-29/CV-analysis-system.git
   cd CV-analysis-system
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR for image processing:
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

4. Create a `.env` file with your API keys:
   ```
   # Copy from example
   cp .env.example .env
   # Edit with your API keys
   nano .env
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app/main.py
   ```

2. Navigate to `http://localhost:8501` in your browser

3. The application has four main sections:
   - **Dashboard**: View statistics about processed CVs
   - **Upload CV**: Upload and process new CV documents
   - **View CVs**: Browse and examine parsed CV information
   - **Chat with CVs**: Query the CV data with natural language

4. To process a CV:
   - Go to the "Upload CV" section
   - Drag and drop or browse for PDF, DOCX, or image files
   - Configure processing options (OCR, LLM usage)
   - Click "Process CVs"

5. To chat with the CV data:
   - Go to the "Chat with CVs" section
   - Enter your query or select a suggested query
   - View the AI-generated response based on the CV data

## Structured Information Extraction

The system extracts the following structured information from CVs:

### Education
- Bachelor's, Master's, and PhD details
- University names, GPAs, majors
- Graduation dates in standardized format

### Work Experience
- Total years of experience
- Company list
- Top responsibilities and projects

### Skills
- Technical skills categorization
- Soft skills identification
- Top skills highlighting

### Personal Information
- Contact details
- Nationality and residence
- Languages spoken

### Employment Analysis
- Current employment status
- Suitable position recommendations
- Candidate rating score (out of 10)

## API Integration

The system integrates with LLM providers (Groq by default) for enhanced parsing and chatbot functionality. To use a different provider:

1. Update the `.env` file with the appropriate API key
2. Modify the provider in the configuration settings:
   ```
   LLM_PROVIDER=your_provider_name
   LLM_MODEL=your_model_name
   ```

## Troubleshooting

- **OCR not working**: Ensure Tesseract is installed and in your PATH or specify the path in `.env`
- **API errors**: Verify your API keys and check provider status
- **Processing errors**: Check file formats and content quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the UI framework
- LangChain for LLM integration
- Tesseract OCR for image processing
- PyMuPDF and python-docx for document handling

