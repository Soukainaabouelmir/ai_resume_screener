# AI Resume Screener

A simple AI-based application that analyzes PDF resumes and predicts how well they match a given job description.

##  Project Overview

This project uses NLP and machine learning techniques to:
- Extract text from PDF resumes
- Preprocess and vectorize the text
- Compare the resume content with a provided job description
- Output a matching score (0 to 100%)

##  Technologies Used

- Python
- PDFPlumber (for PDF parsing)
- Scikit-learn (TF-IDF, cosine similarity)
- Streamlit (for the web interface)
- NLTK / SpaCy (for text preprocessing)
- Pandas & NumPy

##  How to Run Locally

1. **Clone the repository**  
```bash
git clone https://github.com/Soukainaabouelmir/ai_resume_screener.git
cd ai_resume_screener
pip install -r requirements.txt
streamlit run app.py

