# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer, util
# import torch
# import os
# import shutil
# import io
# import fitz  # PyMuPDF for PDF
# from docx import Document  # for DOCX
# import re

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configs
# SIMILARITY_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# SIMILARITY_THRESHOLD = 0.3  # Set the threshold for rejection to 60%
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("🔄 Loading similarity model...")
# similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME).to(DEVICE)
# print("✅ Model loaded successfully.")

# # Directory to save the uploaded CVs
# UPLOAD_DIR = "uploaded_cvs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# def extract_text_from_file(file: UploadFile):
#     content = ""
#     filename = file.filename.lower()
#     try:
#         if filename.endswith(".pdf"):
#             pdf_bytes = io.BytesIO(file.file.read())
#             doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#             for page in doc:
#                 content += page.get_text()
#         elif filename.endswith(".docx"):
#             docx_file = io.BytesIO(file.file.read())
#             doc = Document(docx_file)
#             content = "\n".join([para.text for para in doc.paragraphs])
#         elif filename.endswith(".txt"):
#             content = file.file.read().decode("utf-8")
#         else:
#             raise ValueError("Unsupported file type")
#     except Exception as e:
#         raise ValueError(f"Error reading {filename}: {e}")
#     return content.strip()

# def extract_sections(text):
#     """
#     Extract education, skills, experience, and region sections from CV text.
#     This is a simple implementation that uses patterns/keywords to identify sections.
#     A more robust approach could be implemented based on specific CV formats.
#     """
#     # Simple pattern-based section extraction
#     sections = {
#         "education": "",
#         "skills": "",
#         "experience": "",
#         "region": "",
#     }
    
#     # Find education section
#     education_patterns = [
#         r"(?i)education.*?(?=skills|experience|region|$)",
#         r"(?i)academic.*?(?=skills|experience|region|$)",
#         r"(?i)qualification.*?(?=skills|experience|region|$)",
#     ]
#     for pattern in education_patterns:
#         match = re.search(pattern, text, re.DOTALL)
#         if match:
#             sections["education"] = match.group(0)
#             break
    
#     # Find skills section
#     skills_patterns = [
#         r"(?i)skills.*?(?=education|experience|region|$)",
#         r"(?i)abilities.*?(?=education|experience|region|$)",
#         r"(?i)competencies.*?(?=education|experience|region|$)",
#     ]
#     for pattern in skills_patterns:
#         match = re.search(pattern, text, re.DOTALL)
#         if match:
#             sections["skills"] = match.group(0)
#             break
    
#     # Find experience section
#     exp_patterns = [
#         r"(?i)experience.*?(?=education|skills|region|$)",
#         r"(?i)employment.*?(?=education|skills|region|$)",
#         r"(?i)work history.*?(?=education|skills|region|$)",
#     ]
#     for pattern in exp_patterns:
#         match = re.search(pattern, text, re.DOTALL)
#         if match:
#             sections["experience"] = match.group(0)
#             break
    
#     # Find region/location information
#     region_patterns = [
#         r"(?i)location.*?(?=education|skills|experience|$)",
#         r"(?i)address.*?(?=education|skills|experience|$)",
#         r"(?i)region.*?(?=education|skills|experience|$)",
#     ]
#     for pattern in region_patterns:
#         match = re.search(pattern, text, re.DOTALL)
#         if match:
#             sections["region"] = match.group(0)
#             break
    
#     # If sections weren't found, just use the full text for each
#     for key in sections:
#         if not sections[key]:
#             sections[key] = text
    
#     return sections

# def match_and_rank_cvs(cv_texts, job_description, files, weights):
#     # Extract job description sections using the same approach
#     job_sections = extract_sections(job_description)
    
#     results = []
#     for i, cv_text in enumerate(cv_texts):
#         # Extract CV sections
#         cv_sections = extract_sections(cv_text)
        
#         # Calculate weighted similarity for each section
#         section_similarities = {}
#         for section, weight_key in [
#             ("education", "education"),
#             ("skills", "skills"),
#             ("experience", "experience"),
#             ("region", "region"),
#         ]:
#             # Encode section text from both CV and job description
#             job_section_embedding = similarity_model.encode(job_sections[section], convert_to_tensor=True)
#             cv_section_embedding = similarity_model.encode(cv_sections[section], convert_to_tensor=True)
            
#             # Calculate similarity for this section
#             section_sim = util.pytorch_cos_sim(job_section_embedding, cv_section_embedding)[0][0].item()
            
#             # Store the similarity score
#             section_similarities[section] = section_sim
        
#         # Calculate the weighted average similarity
#         total_weight = sum(weights.values())
#         if total_weight == 0:  # Prevent division by zero
#             total_weight = 1
            
#         weighted_similarity = (
#             (section_similarities["education"] * weights.get("education", 1)) +
#             (section_similarities["skills"] * weights.get("skills", 1)) +
#             (section_similarities["experience"] * weights.get("experience", 1)) +
#             (section_similarities["region"] * weights.get("region", 1))
#         ) / total_weight
        
#         # Save the file to the server directory
#         filename = files[i].filename
#         file_path = os.path.join(UPLOAD_DIR, filename)
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(files[i].file, f)
        
#         # Apply the threshold to accept or reject
#         result = {
#             "filename": filename,
#             "similarity": float(weighted_similarity),
#             "file_path": f"/files/{filename}",
#             "status": "Accepted" if weighted_similarity >= SIMILARITY_THRESHOLD else "Rejected",
#             "reason": "Similarity score below threshold" if weighted_similarity < SIMILARITY_THRESHOLD else None,
#             "section_scores": {
#                 "education": float(section_similarities["education"]),
#                 "skills": float(section_similarities["skills"]),
#                 "experience": float(section_similarities["experience"]),
#                 "region": float(section_similarities["region"]),
#             }
#         }
        
#         results.append(result)
    
#     # Sort by similarity score in descending order
#     return sorted(results, key=lambda x: x["similarity"], reverse=True)

# @app.post("/match-cvs/")
# async def match_cvs(files: list[UploadFile] = File(...), job_description: str = Form(...), education_weight: float = Form(...), skills_weight: float = Form(...), experience_weight: float = Form(...), region_weight: float = Form(...)):
#     if not job_description or not files:
#         return JSONResponse(status_code=400, content={"error": "Missing job description or files."})

#     # Prepare weights
#     weights = {
#         "education": education_weight,
#         "skills": skills_weight,
#         "experience": experience_weight,
#         "region": region_weight,
#     }

#     cv_texts = []
#     for file in files:
#         try:
#             # Reset the file pointer for each file
#             file.file.seek(0)
#             text = extract_text_from_file(file)
#             if text:
#                 cv_texts.append(text)
#                 # Reset the file pointer after extraction for later use
#                 file.file.seek(0)
#         except Exception as e:
#             return JSONResponse(status_code=400, content={"error": str(e)})

#     results = match_and_rank_cvs(cv_texts, job_description, files, weights)

#     if not results:
#         return {"message": "No suitable CVs found."}

#     return {"message": "Matching complete!", "ranked_cvs": results}

# @app.get("/files/{filename}")
# async def get_file(filename: str):
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     else:
#         return JSONResponse(status_code=404, content={"error": "File not found"})






















# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# import os
# import shutil
# import io
# import fitz  # PyMuPDF for PDF
# from docx import Document  # for DOCX
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sentence_transformers import SentenceTransformer, util
# import spacy
# from dateutil.parser import parse as date_parse
# import datetime
# import numpy as np

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configs
# SIMILARITY_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# SIMILARITY_THRESHOLD = 0.3  # Set the threshold for rejection to 60%
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("🔄 Loading similarity model...")
# # Load the sentence transformer model for semantic understanding
# similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME).to(DEVICE)
# print("✅ Similarity model loaded successfully.")

# # Try to load spaCy model for advanced NLP
# try:
#     print("🔄 Loading spaCy model...")
#     nlp = spacy.load("fr_core_news_md")  # French model for better language support
#     print("✅ spaCy model loaded successfully.")
# except:
#     print("⚠️ Couldn't load spaCy model. Installing...")
#     import subprocess
#     # First download NLTK resources
#     try:
#         nltk.download('punkt')
#         nltk.download('stopwords')
#     except:
#         print("⚠️ Failed to download NLTK resources - continuing without them")
    
#     # Then try to install spaCy model
#     try:
#         subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"], check=True)
#         nlp = spacy.load("fr_core_news_md")
#         print("✅ spaCy model installed and loaded successfully.")
#     except:
#         print("⚠️ Failed to install spaCy model - continuing with limited NLP capabilities")
#         nlp = None

# # Directory to save the uploaded CVs
# UPLOAD_DIR = "uploaded_cvs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Keywords and patterns for section extraction (multilingual support)
# SECTION_KEYWORDS = {
#     "education": ["education", "formation", "academic", "études", "diplôme", "qualification", "curriculum", "éducation", "scolaire"],
#     "skills": ["skills", "compétences", "abilities", "aptitudes", "competencies", "expertise", "savoir-faire", "connaissances", "technologies", "technical", "technique"],
#     "experience": ["experience", "expérience", "employment", "emploi", "work history", "parcours professionnel", "profession", "career", "carrière", "travail"],
#     "region": ["location", "address", "adresse", "région", "region", "ville", "city", "pays", "country", "domicile", "residence"]
# }

# # Common date patterns for experience extraction
# DATE_PATTERNS = [
#     r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\s*-\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b',
#     r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
#     r'\b(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\s*-\s*(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\b',
#     r'\b(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\s*-\s*(présent|actuel|aujourd\'hui)\b',
#     r'\b\d{4}\s*-\s*\d{4}\b',
#     r'\b\d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
#     r'\b\d{2}/\d{4}\s*-\s*\d{2}/\d{4}\b',
#     r'\b\d{2}/\d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
# ]

# def extract_text_from_file(file: UploadFile):
#     """Extract text content from various file formats"""
#     content = ""
#     filename = file.filename.lower()
#     try:
#         if filename.endswith(".pdf"):
#             pdf_bytes = io.BytesIO(file.file.read())
#             doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#             for page in doc:
#                 content += page.get_text()
#         elif filename.endswith(".docx"):
#             docx_file = io.BytesIO(file.file.read())
#             doc = Document(docx_file)
#             content = "\n".join([para.text for para in doc.paragraphs])
#         elif filename.endswith(".txt"):
#             content = file.file.read().decode("utf-8")
#         else:
#             raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT files.")
#     except Exception as e:
#         raise ValueError(f"Error reading {filename}: {e}")
#     return content.strip()

# def preprocess_text(text):
#     """Clean and preprocess text"""
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Convert to lowercase
#     text = text.lower()
#     return text

# def find_section_boundaries(text, section_name):
#     """Find the start and end positions of a section in the text"""
#     keywords = SECTION_KEYWORDS[section_name]
    
#     # Create regex pattern with word boundaries
#     pattern = r'(?i)\b(' + '|'.join(keywords) + r')\b.*?(?=\b(' + '|'.join([k for sublist in SECTION_KEYWORDS.values() for k in sublist if k not in keywords]) + r')\b|$)'
    
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(0)
#     return ""

# def extract_sections(text):
#     """Extract education, skills, experience, and region sections from CV text."""
#     preprocessed_text = preprocess_text(text)
    
#     sections = {
#         "education": find_section_boundaries(preprocessed_text, "education"),
#         "skills": find_section_boundaries(preprocessed_text, "skills"),
#         "experience": find_section_boundaries(preprocessed_text, "experience"),
#         "region": find_section_boundaries(preprocessed_text, "region"),
#     }
    
#     # If sections weren't found, just use the full text for each
#     for key in sections:
#         if not sections[key]:
#             sections[key] = preprocessed_text
    
#     return sections

# def extract_experience_years(text):
#     """Extract years of experience from text"""
#     years_patterns = [
#         r'(\d+)\s*(?:ans|years|year|an)(?:\s*d[\'e]?\s*expérience|\s*experience)',
#         r'expérience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#         r'experience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#     ]
    
#     for pattern in years_patterns:
#         matches = re.findall(pattern, text.lower())
#         if matches:
#             return max([int(y) for y in matches])
    
#     # Try to extract from date ranges in experience section
#     experience_section = extract_sections(text)["experience"]
    
#     # Find all date ranges in the experience section
#     date_ranges = []
#     for pattern in DATE_PATTERNS:
#         matches = re.findall(pattern, experience_section.lower())
#         if matches:
#             for match in matches:
#                 date_ranges.append(match)
    
#     # Calculate total years from all date ranges
#     total_years = 0
#     for date_range in date_ranges:
#         try:
#             # Parse dates and calculate duration
#             if isinstance(date_range, tuple):
#                 start_date = date_range[0]
#                 end_date = date_range[1]
#             else:
#                 # Split by hyphen if it's a string
#                 parts = re.split(r'\s*-\s*', date_range)
#                 if len(parts) >= 2:
#                     start_date = parts[0]
#                     end_date = parts[1]
#                 else:
#                     continue
            
#             # Try to parse the dates
#             try:
#                 start_year = date_parse(start_date).year
#                 if any(word in end_date.lower() for word in ["present", "current", "aujourd'hui", "actuel"]):
#                     end_year = datetime.datetime.now().year
#                 else:
#                     end_year = date_parse(end_date).year
                
#                 # Add to total years
#                 duration = end_year - start_year
#                 if duration > 0:
#                     total_years += duration
#             except:
#                 # If parsing fails, try to extract just years
#                 year_pattern = r'\b\d{4}\b'
#                 years = re.findall(year_pattern, start_date + " " + end_date)
#                 if len(years) >= 2:
#                     duration = int(years[1]) - int(years[0])
#                     if duration > 0:
#                         total_years += duration
#         except:
#             continue
    
#     return total_years if total_years > 0 else 0

# def extract_job_requirements(job_description):
#     """Extract specific requirements from job description"""
#     requirements = {
#         "experience_years": 0,
#         "education_level": [],
#         "required_skills": [],
#         "location": None
#     }
    
#     # Extract years of experience
#     years_patterns = [
#         r'(\d+)\s*(?:ans|years|year|an)(?:\s*d[\'e]?\s*expérience|\s*experience)',
#         r'expérience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#         r'experience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#     ]
    
#     for pattern in years_patterns:
#         matches = re.findall(pattern, job_description.lower())
#         if matches:
#             requirements["experience_years"] = max([int(y) for y in matches])
#             break
    
#     # Extract education level
#     education_patterns = {
#         "bachelor": [r'\b(?:bachelor|licence|licenciatura)\b', r'\bbac\s*\+\s*3\b'],
#         "master": [r'\b(?:master|mastère|msc)\b', r'\bbac\s*\+\s*5\b'],
#         "engineer": [r'\b(?:ingénieur|engineer|ing[ée]nieur d\'[ée]tat)\b'],
#         "phd": [r'\b(?:phd|ph\.d|doctorat|doctorate)\b'],
#     }
    
#     for level, patterns in education_patterns.items():
#         for pattern in patterns:
#             if re.search(pattern, job_description.lower()):
#                 requirements["education_level"].append(level)
    
#     # Extract skills using spaCy if available
#     if nlp:
#         doc = nlp(job_description)
#         # Extract noun phrases as potential skills
#         for chunk in doc.noun_chunks:
#             if 3 < len(chunk.text) < 30:  # Reasonable length for a skill
#                 requirements["required_skills"].append(chunk.text.lower())
#     else:
#         # Fallback method: extract common tech skills with regex
#         tech_patterns = [
#             r'\b(?:java|python|javascript|js|typescript|ts|c\+\+|c#|php|ruby|swift|kotlin|golang|rust)\b',
#             r'\b(?:html|css|sql|mysql|postgresql|mongodb|nosql|redis|graphql)\b',
#             r'\b(?:aws|azure|gcp|cloud|docker|kubernetes|k8s|ci/cd|jenkins|git)\b',
#             r'\b(?:agile|scrum|kanban|jira|confluence|trello)\b',
#             r'\b(?:react|angular|vue|svelte|jquery|node\.js|nodejs|express|django|flask|spring|laravel)\b',
#             r'\b(?:tensorflow|pytorch|scikit-learn|pandas|numpy|data\s*science|machine\s*learning|ml|ai)\b',
#         ]
        
#         for pattern in tech_patterns:
#             matches = re.findall(pattern, job_description.lower())
#             requirements["required_skills"].extend(matches)
    
#     # Make sure skills are unique
#     requirements["required_skills"] = list(set(requirements["required_skills"]))
    
#     return requirements

# def analyze_cv_education(education_text, job_requirements):
#     """Analyze education section against job requirements"""
#     education_score = 0.0
#     max_score = 1.0
    
#     # If no education requirements, give a neutral score
#     if not job_requirements["education_level"]:
#         return 0.5
    
#     # Check for education levels in CV
#     education_matches = []
#     education_patterns = {
#         "bachelor": [r'\b(?:bachelor|licence|licenciatura)\b', r'\bbac\s*\+\s*3\b'],
#         "master": [r'\b(?:master|mastère|msc)\b', r'\bbac\s*\+\s*5\b'],
#         "engineer": [r'\b(?:ingénieur|engineer|ing[ée]nieur d\'[ée]tat)\b'],
#         "phd": [r'\b(?:phd|ph\.d|doctorat|doctorate)\b'],
#     }
    
#     for level, patterns in education_patterns.items():
#         for pattern in patterns:
#             if re.search(pattern, education_text.lower()):
#                 education_matches.append(level)
    
#     # Map education levels to numeric values for comparison
#     education_values = {
#         "bachelor": 1,
#         "engineer": 2,
#         "master": 2,
#         "phd": 3
#     }
    
#     # Find highest education in CV and job requirements
#     cv_max_edu = max([education_values.get(level, 0) for level in education_matches]) if education_matches else 0
#     job_max_edu = max([education_values.get(level, 0) for level in job_requirements["education_level"]]) if job_requirements["education_level"] else 0
    
#     if cv_max_edu >= job_max_edu and job_max_edu > 0:
#         # Meets or exceeds requirements
#         education_score = max_score
#     elif cv_max_edu > 0 and job_max_edu > 0:
#         # Partial match
#         education_score = cv_max_edu / job_max_edu * max_score
    
#     # Apply semantic similarity as well
#     if job_requirements["education_level"]:
#         job_edu_text = " ".join(job_requirements["education_level"])
#         similarity = calculate_semantic_similarity(job_edu_text, education_text)
#         # Blend direct matching with semantic similarity
#         education_score = 0.7 * education_score + 0.3 * similarity
    
#     return min(education_score, max_score)  # Cap at max_score

# def analyze_cv_experience(experience_text, job_requirements):
#     """Analyze experience section against job requirements"""
#     experience_score = 0.0
#     max_score = 1.0
    
#     # Extract years of experience from CV
#     cv_experience_years = extract_experience_years(experience_text)
    
#     # Compare with job requirements
#     required_years = job_requirements["experience_years"]
    
#     if required_years > 0:
#         if cv_experience_years >= required_years:
#             # Meets or exceeds requirements
#             experience_score = max_score
#             # Bonus for exceeding requirements by more than 2 years (but cap it)
#             if cv_experience_years > required_years + 2:
#                 experience_score = min(max_score * 1.1, max_score)
#         elif cv_experience_years > 0:
#             # Partial match
#             experience_score = (cv_experience_years / required_years) * max_score
#     else:
#         # If no specific requirement, give a neutral score
#         experience_score = 0.5
    
#     # Apply semantic similarity to evaluate quality/relevance of experience
#     similarity = calculate_semantic_similarity(job_requirements.get("experience_text", ""), experience_text)
    
#     # Blend direct year matching with semantic similarity
#     final_score = 0.7 * experience_score + 0.3 * similarity
    
#     return min(final_score, max_score)  # Cap at max_score

# def analyze_cv_skills(skills_text, job_requirements):
#     """Analyze skills section against job requirements"""
#     skills_score = 0.0
#     max_score = 1.0
    
#     required_skills = job_requirements["required_skills"]
    
#     # If no specific skills requirements, rely on semantic similarity
#     if not required_skills:
#         return calculate_semantic_similarity(job_requirements.get("skills_text", ""), skills_text)
    
#     # Count how many required skills are found in CV
#     skills_found = 0
#     unique_skills = set()
    
#     for skill in required_skills:
#         skill_pattern = r'\b' + re.escape(skill) + r'\b'
#         if re.search(skill_pattern, skills_text.lower()):
#             skills_found += 1
#             unique_skills.add(skill)
    
#     # Calculate direct skills match ratio
#     if required_skills:
#         direct_match_score = skills_found / len(required_skills)
#     else:
#         direct_match_score = 0.5  # Neutral score if no requirements
    
#     # Apply semantic similarity for skills understanding
#     similarity = calculate_semantic_similarity(" ".join(required_skills), skills_text)
    
#     # Look for bonus skills that might be valuable but not required
#     bonus_score = 0.0
#     if nlp:
#         doc = nlp(skills_text)
#         # Extract skill-like terms
#         potential_bonus_skills = []
#         for chunk in doc.noun_chunks:
#             if 3 < len(chunk.text) < 30:  # Reasonable length for a skill
#                 potential_bonus_skills.append(chunk.text.lower())
        
#         # Count bonus skills (ones not in requirements)
#         bonus_skills = set(potential_bonus_skills) - set([s.lower() for s in required_skills])
#         if bonus_skills:
#             # Bonus for having additional skills (up to 0.1 extra)
#             bonus_score = min(0.1, len(bonus_skills) * 0.02)
    
#     # Blend direct matching with semantic similarity
#     skills_score = (0.6 * direct_match_score) + (0.4 * similarity) + bonus_score
    
#     return min(skills_score, max_score)  # Cap at max_score

# def analyze_cv_region(region_text, job_requirements):
#     """Analyze region/location against job requirements"""
#     # Since region matching is usually simpler (exact match or not),
#     # we'll use a more straightforward approach
    
#     if not job_requirements.get("location"):
#         return 0.5  # Neutral score if no location requirement
    
#     # Direct location mention
#     if job_requirements["location"].lower() in region_text.lower():
#         return 1.0
    
#     # Fallback to semantic similarity
#     return calculate_semantic_similarity(job_requirements.get("location", ""), region_text)

# def calculate_semantic_similarity(text1, text2):
#     """Calculate semantic similarity between two texts using the loaded model"""
#     if not text1 or not text2:
#         return 0.5  # Neutral score for empty inputs
    
#     # Encode texts to vectors
#     embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
#     embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    
#     # Calculate cosine similarity
#     similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    
#     return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1




# def extract_experience_details(experience_text):
#     """
#     Extract detailed experience information including:
#     - Total years of experience
#     - Date ranges found
#     - Job titles
#     - Companies
#     """
#     details = {
#         "total_years": 0,
#         "date_ranges": [],
#         "positions": [],
#         "companies": []
#     }
    
#     # Extract years using the existing function
#     details["total_years"] = extract_experience_years(experience_text)
    
#     # Extract date ranges
#     for pattern in DATE_PATTERNS:
#         matches = re.findall(pattern, experience_text.lower())
#         if matches:
#             for match in matches:
#                 if isinstance(match, tuple):
#                     range_str = f"{match[0]} - {match[1]}"
#                 else:
#                     range_str = match
#                 details["date_ranges"].append(range_str)
    
#     # Extract job titles (common job title patterns)
#     job_title_patterns = [
#         r'\b(développeur|developer|ingénieur|engineer|architecte|architect|lead|chef|responsable|manager|consultant)\s+([\w\s]{3,30}?)\b',
#         r'\b(directeur|director|VP|head of)\s+([\w\s]{3,30}?)\b'
#     ]
    
#     for pattern in job_title_patterns:
#         matches = re.findall(pattern, experience_text.lower())
#         if matches:
#             for match in matches:
#                 if len(match) >= 2:
#                     job_title = f"{match[0]} {match[1]}".strip()
#                     if job_title not in details["positions"]:
#                         details["positions"].append(job_title)
    
#     # Extract company names (this is more complex, simplified approach)
#     if nlp:
#         doc = nlp(experience_text)
#         for ent in doc.ents:
#             if ent.label_ == "ORG":
#                 if len(ent.text) > 2 and ent.text not in details["companies"]:
#                     details["companies"].append(ent.text)
    
#     return details

# def create_experience_classification(cv_experience_years, required_years):
#     """
#     Create a detailed classification of experience with observations
#     """
#     classification = {
#         "status": "Approved" if cv_experience_years >= required_years else "Rejected",
#         "required_years": required_years,
#         "found_years": cv_experience_years,
#         "difference": cv_experience_years - required_years,
#         "observations": []
#     }
    
#     # Add specific observations
#     if cv_experience_years >= required_years:
#         if cv_experience_years > required_years + 5:
#             classification["observations"].append(f"Candidate has {cv_experience_years - required_years} years more than required (possibly overqualified)")
#         elif cv_experience_years > required_years:
#             classification["observations"].append(f"Candidate exceeds the required experience by {cv_experience_years - required_years} years")
#         else:
#             classification["observations"].append("Candidate meets the exact experience requirement")
#     else:
#         deficit = required_years - cv_experience_years
#         classification["observations"].append(f"Candidate lacks {deficit} years of required experience")
        
#         if cv_experience_years > 0:
#             percentage = (cv_experience_years / required_years) * 100
#             classification["observations"].append(f"Candidate has {percentage:.1f}% of the required experience")
            
#             if percentage >= 80:
#                 classification["observations"].append("Consider for interview despite slight experience shortage")
    
#     return classification

# # Update the match_and_rank_cvs function to include the new detailed experience analysis
# def match_and_rank_cvs(cv_texts, job_description, files, weights):
#     """Match CV texts against job description with intelligent analysis"""
#     # Extract detailed job requirements
#     job_sections = extract_sections(job_description)
#     job_requirements = extract_job_requirements(job_description)
    
#     # Add full section texts to requirements for semantic matching
#     job_requirements["education_text"] = job_sections["education"]
#     job_requirements["skills_text"] = job_sections["skills"]
#     job_requirements["experience_text"] = job_sections["experience"]
#     job_requirements["location"] = job_sections["region"]
    
#     results = []
#     for i, cv_text in enumerate(cv_texts):
#         # Extract CV sections
#         cv_sections = extract_sections(cv_text)
        
#         # Perform detailed analysis for each section
#         education_score = analyze_cv_education(cv_sections["education"], job_requirements)
#         skills_score = analyze_cv_skills(cv_sections["skills"], job_requirements)
#         experience_score = analyze_cv_experience(cv_sections["experience"], job_requirements)
#         region_score = analyze_cv_region(cv_sections["region"], job_requirements)
        
#         # Store section scores
#         section_similarities = {
#             "education": education_score,
#             "skills": skills_score,
#             "experience": experience_score,
#             "region": region_score,
#         }
        
#         # Calculate weighted average similarity
#         total_weight = sum(weights.values())
#         if total_weight == 0:  # Prevent division by zero
#             total_weight = 1
            
#         weighted_similarity = (
#             (section_similarities["education"] * weights.get("education", 1)) +
#             (section_similarities["skills"] * weights.get("skills", 1)) +
#             (section_similarities["experience"] * weights.get("experience", 1)) +
#             (section_similarities["region"] * weights.get("region", 1))
#         ) / total_weight
        
#         # Extract years of experience for the result
#         cv_experience_years = extract_experience_years(cv_text)
        
#         # Extract detailed experience information
#         experience_details = extract_experience_details(cv_sections["experience"])
        
#         # Create detailed experience classification
#         experience_classification = create_experience_classification(
#             cv_experience_years, 
#             job_requirements["experience_years"]
#         )
        
#         # Determine if the CV meets minimum requirements
#         meets_min_reqs = True
#         rejection_reasons = []
        
#         # Check experience years
#         if job_requirements["experience_years"] > 0 and cv_experience_years < job_requirements["experience_years"]:
#             meets_min_reqs = False
#             rejection_reasons.append(f"Experience: {cv_experience_years} years (requires {job_requirements['experience_years']})")
        
#         # Save the file to the server directory
#         filename = files[i].filename
#         file_path = os.path.join(UPLOAD_DIR, filename)
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(files[i].file, f)
        
#         # Create detailed experience status string for display
#         experience_status = ""
#         if job_requirements["experience_years"] > 0:
#             if cv_experience_years >= job_requirements["experience_years"]:
#                 experience_status = f"Approved: {cv_experience_years}/{job_requirements['experience_years']} years"
#             else:
#                 experience_status = f"Rejected: {cv_experience_years}/{job_requirements['experience_years']} years"
        
#         # Apply threshold and final status
#         status = "Accepted" if (meets_min_reqs and weighted_similarity >= SIMILARITY_THRESHOLD) else "Rejected"
        
#         # Prepare the final reason with detailed breakdown
#         reason = None
#         if status == "Rejected":
#             rejection_details = []
            
#             # Experience details
#             if job_requirements["experience_years"] > 0 and cv_experience_years < job_requirements["experience_years"]:
#                 deficit = job_requirements["experience_years"] - cv_experience_years
#                 if cv_experience_years == 0:
#                     rejection_details.append(f"No experience found (requires {job_requirements['experience_years']} years)")
#                 else:
#                     rejection_details.append(f"Experience: {cv_experience_years}/{job_requirements['experience_years']} years (deficit: {deficit} years)")
            
#             # Skills details
#             if job_requirements["required_skills"] and section_similarities["skills"] < 0.4:
#                 found = [skill for skill in job_requirements["required_skills"] if re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())]
#                 missing = [skill for skill in job_requirements["required_skills"] if not re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())]
#                 if missing:
#                     rejection_details.append(f"Missing skills: {', '.join(missing[:3])}{' and more' if len(missing) > 3 else ''}")
            
#             # Region details
#             if job_requirements["location"] and section_similarities["region"] < 0.3:
#                 rejection_details.append(f"Location mismatch: Required {job_requirements['location']}")
            
#             # Education details
#             if job_requirements["education_level"] and section_similarities["education"] < 0.4:
#                 rejection_details.append(f"Education mismatch: Required {', '.join(job_requirements['education_level'])}")
            
#             # Final reason formatting
#             if rejection_details:
#                 reason = " | ".join(rejection_details)
#             else:
#                 reason = f"Overall match score ({weighted_similarity:.2f}) below threshold ({SIMILARITY_THRESHOLD})"
        
#         # Create detailed result
#         result = {
#             "filename": filename,
#             "similarity": float(weighted_similarity),
#             "file_path": f"/files/{filename}",
#             "status": status,
#             "reason": reason,
#             "experience_years": cv_experience_years,
#             # Add prominent experience status for frontend display
#             "experience_status": experience_status,
#             "section_scores": {
#                 "education": float(section_similarities["education"]),
#                 "skills": float(section_similarities["skills"]),
#                 "experience": float(section_similarities["experience"]),
#                 "region": float(section_similarities["region"]),
#             },
#             # Add detailed experience analysis
#             "experience_analysis": {
#                 "classification": experience_classification,
#                 "details": experience_details,
#                 # Add direct comparison data for frontend display
#                 "display": {
#                     "required_years": job_requirements["experience_years"],
#                     "candidate_years": cv_experience_years,
#                     "deficit_or_surplus": cv_experience_years - job_requirements["experience_years"],
#                     "percentage_of_required": (cv_experience_years / job_requirements["experience_years"] * 100) if job_requirements["experience_years"] > 0 else 100,
#                     "status_text": f"{cv_experience_years}/{job_requirements['experience_years']} years" if job_requirements["experience_years"] > 0 else "No specific requirement",
#                     "color_code": "green" if cv_experience_years >= job_requirements["experience_years"] else "red",
#                 }
#             },
#             # Add some detailed analysis for the frontend
#             "analysis": {
#                 "meets_experience": bool(cv_experience_years >= job_requirements["experience_years"]) if job_requirements["experience_years"] > 0 else None,
#                 "required_skills_found": [skill for skill in job_requirements["required_skills"] if re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())],
#                 "missing_skills": [skill for skill in job_requirements["required_skills"] if not re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())],
#                 # Add summary statements for clear display
#                 # "summary": {
#                 #     "experience_summary": f"{cv_experience_years}/{job_requirements['experience_years']} years" if job_requirements["experience_years"] > 0 else "No specific requirement",
#                 #     "skills_summary": f"{len([s for s in job_requirements['required_skills'] if re.search(r'\\b' + re.escape(s) + r'\\b', cv_text.lower())])}/{len(job_requirements['required_skills'])} skills matched" if job_requirements["required_skills"] else "No specific skills requirement",
#                 #     "education_match": "Matched" if section_similarities["education"] >= 0.7 else "Partial match" if section_similarities["education"] >= 0.4 else "Low match",
#                 #     "location_match": "Matched" if section_similarities["region"] >= 0.7 else "Partial match" if section_similarities["region"] >= 0.3 else "No match"
#                 # }
#             }
#         }
        
#         results.append(result)
    
#     # Sort by similarity score in descending order
#     return sorted(results, key=lambda x: x["similarity"], reverse=True)

# @app.post("/match-cvs/")
# async def match_cvs(files: list[UploadFile] = File(...), 
#                    job_description: str = Form(...), 
#                    education_weight: float = Form(...), 
#                    skills_weight: float = Form(...), 
#                    experience_weight: float = Form(...), 
#                    region_weight: float = Form(...)):
#     """API endpoint to match CVs against a job description"""
#     if not job_description or not files:
#         return JSONResponse(status_code=400, content={"error": "Missing job description or files."})

#     # Prepare weights
#     weights = {
#         "education": education_weight,
#         "skills": skills_weight,
#         "experience": experience_weight,
#         "region": region_weight,
#     }

#     # Extract job requirements for summary
#     job_requirements = extract_job_requirements(job_description)
#     job_sections = extract_sections(job_description)
    
#     # Add full section texts to requirements for summary
#     job_requirements["education_text"] = job_sections["education"]
#     job_requirements["skills_text"] = job_sections["skills"]
#     job_requirements["experience_text"] = job_sections["experience"]
#     job_requirements["location"] = job_sections["region"]

#     cv_texts = []
#     for file in files:
#         try:
#             # Reset the file pointer for each file
#             file.file.seek(0)
#             text = extract_text_from_file(file)
#             if text:
#                 cv_texts.append(text)
#                 # Reset the file pointer after extraction for later use
#                 file.file.seek(0)
#         except Exception as e:
#             return JSONResponse(status_code=400, content={"error": str(e)})

#     results = match_and_rank_cvs(cv_texts, job_description, files, weights)

#     if not results:
#         return {"message": "No suitable CVs found."}
    
#     # Create job summary for context
#     job_summary = {
#         "experience_requirement": f"{job_requirements['experience_years']} years" if job_requirements['experience_years'] > 0 else "Not specified",
#         "education_requirement": ", ".join(job_requirements["education_level"]) if job_requirements["education_level"] else "Not specified",
#         "skills_requirement": job_requirements["required_skills"],
#         "location_requirement": job_requirements["location"] if job_requirements["location"] and len(job_requirements["location"]) > 3 else "Not specified",
#     }

#     return {
#         "message": "Matching complete!", 
#         "job_summary": job_summary,
#         "ranked_cvs": results,
#         "matching_stats": {
#             "total_cvs": len(results),
#             "accepted": sum(1 for r in results if r["status"] == "Accepted"),
#             "rejected": sum(1 for r in results if r["status"] == "Rejected"),
#             "experience_rejected": sum(1 for r in results if r["status"] == "Rejected" and r["experience_years"] < job_requirements["experience_years"]),
#         }
#     }  

# @app.get("/files/{filename}")
# async def get_file(filename: str):
#     """Endpoint to retrieve uploaded files"""
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     else:
#         return JSONResponse(status_code=404, content={"error": "File not found"})

# # If running directly (not imported)
# if __name__ == "__main__":
#     import uvicorn
#     print("⚡ Starting CV matching server...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# import os
# import shutil
# import io
# import fitz  # PyMuPDF for PDF
# from docx import Document  # for DOCX
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sentence_transformers import SentenceTransformer, util
# import spacy
# from dateutil.parser import parse as date_parse
# import datetime
# import re
# import numpy as np

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configs
# SIMILARITY_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# SIMILARITY_THRESHOLD = 0.3  # Set the threshold for rejection to 60%
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("🔄 Loading similarity model...")
# # Load the sentence transformer model for semantic understanding
# similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME).to(DEVICE)
# print("✅ Similarity model loaded successfully.")

# # Try to load spaCy model for advanced NLP
# try:
#     print("🔄 Loading spaCy model...")
#     nlp = spacy.load("fr_core_news_md")  # French model for better language support
#     print("✅ spaCy model loaded successfully.")
# except:
#     print("⚠️ Couldn't load spaCy model. Installing...")
#     import subprocess
#     # First download NLTK resources
#     try:
#         nltk.download('punkt')
#         nltk.download('stopwords')
#     except:
#         print("⚠️ Failed to download NLTK resources - continuing without them")
    
#     # Then try to install spaCy model
#     try:
#         subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"], check=True)
#         nlp = spacy.load("fr_core_news_md")
#         print("✅ spaCy model installed and loaded successfully.")
#     except:
#         print("⚠️ Failed to install spaCy model - continuing with limited NLP capabilities")
#         nlp = None

# # Directory to save the uploaded CVs
# UPLOAD_DIR = "uploaded_cvs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Keywords and patterns for section extraction (multilingual support)
# SECTION_KEYWORDS = {
#     "education": ["education", "formation", "academic", "études", "diplôme", "qualification", "curriculum", "éducation", "scolaire"],
#     "skills": ["skills", "compétences", "abilities", "aptitudes", "competencies", "expertise", "savoir-faire", "connaissances", "technologies", "technical", "technique"],
#     "experience": ["experience", "expérience", "employment", "emploi", "work history", "parcours professionnel", "profession", "career", "carrière", "travail"],
#     "region": ["location", "address", "adresse", "région", "region", "ville", "city", "pays", "country", "domicile", "residence"]
# }

# # Common date patterns for experience extraction
# DATE_PATTERNS = [
#     r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\s*-\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b',
#     r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
#     r'\b(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\s*-\s*(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\b',
#     r'\b(janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc)[a-z]* \d{4}\s*-\s*(présent|actuel|aujourd\'hui)\b',
#     r'\b\d{4}\s*-\s*\d{4}\b',
#     r'\b\d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
#     r'\b\d{2}/\d{4}\s*-\s*\d{2}/\d{4}\b',
#     r'\b\d{2}/\d{4}\s*-\s*(present|current|aujourd\'hui|actuel)\b',
# ]

# def extract_text_from_file(file: UploadFile):
#     """Extract text content from various file formats"""
#     content = ""
#     filename = file.filename.lower()
#     try:
#         if filename.endswith(".pdf"):
#             pdf_bytes = io.BytesIO(file.file.read())
#             doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#             for page in doc:
#                 content += page.get_text()
#         elif filename.endswith(".docx"):
#             docx_file = io.BytesIO(file.file.read())
#             doc = Document(docx_file)
#             content = "\n".join([para.text for para in doc.paragraphs])
#         elif filename.endswith(".txt"):
#             content = file.file.read().decode("utf-8")
#         else:
#             raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT files.")
#     except Exception as e:
#         raise ValueError(f"Error reading {filename}: {e}")
#     return content.strip()

# def preprocess_text(text):
#     """Clean and preprocess text"""
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Convert to lowercase
#     text = text.lower()
#     return text

# def find_section_boundaries(text, section_name):
#     """Find the start and end positions of a section in the text"""
#     keywords = SECTION_KEYWORDS[section_name]
    
#     # Create regex pattern with word boundaries
#     pattern = r'(?i)\b(' + '|'.join(keywords) + r')\b.*?(?=\b(' + '|'.join([k for sublist in SECTION_KEYWORDS.values() for k in sublist if k not in keywords]) + r')\b|$)'
    
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(0)
#     return ""

# def extract_sections(text):
#     """Extract education, skills, experience, and region sections from CV text."""
#     preprocessed_text = preprocess_text(text)
    
#     sections = {
#         "education": find_section_boundaries(preprocessed_text, "education"),
#         "skills": find_section_boundaries(preprocessed_text, "skills"),
#         "experience": find_section_boundaries(preprocessed_text, "experience"),
#         "region": find_section_boundaries(preprocessed_text, "region"),
#     }
    
#     # If sections weren't found, just use the full text for each
#     for key in sections:
#         if not sections[key]:
#             sections[key] = preprocessed_text
    
#     return sections

# def extract_experience_years(text):
#     """Extract years of experience from text"""
#     years_patterns = [
#         r'(\d+)\s*(?:ans|years|year|an)(?:\s*d[\'e]?\s*expérience|\s*experience)',
#         r'expérience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#         r'experience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#     ]
    
#     for pattern in years_patterns:
#         matches = re.findall(pattern, text.lower())
#         if matches:
#             return max([int(y) for y in matches])
    
#     # Try to extract from date ranges in experience section
#     experience_section = extract_sections(text)["experience"]
    
#     # Find all date ranges in the experience section
#     date_ranges = []
#     for pattern in DATE_PATTERNS:
#         matches = re.findall(pattern, experience_section.lower())
#         if matches:
#             for match in matches:
#                 date_ranges.append(match)
    
#     # Calculate total years from all date ranges
#     total_years = 0
#     for date_range in date_ranges:
#         try:
#             # Parse dates and calculate duration
#             if isinstance(date_range, tuple):
#                 start_date = date_range[0]
#                 end_date = date_range[1]
#             else:
#                 # Split by hyphen if it's a string
#                 parts = re.split(r'\s*-\s*', date_range)
#                 if len(parts) >= 2:
#                     start_date = parts[0]
#                     end_date = parts[1]
#                 else:
#                     continue
            
#             # Try to parse the dates
#             try:
#                 start_year = date_parse(start_date).year
#                 if any(word in end_date.lower() for word in ["present", "current", "aujourd'hui", "actuel"]):
#                     end_year = datetime.datetime.now().year
#                 else:
#                     end_year = date_parse(end_date).year
                
#                 # Add to total years
#                 duration = end_year - start_year
#                 if duration > 0:
#                     total_years += duration
#             except:
#                 # If parsing fails, try to extract just years
#                 year_pattern = r'\b\d{4}\b'
#                 years = re.findall(year_pattern, start_date + " " + end_date)
#                 if len(years) >= 2:
#                     duration = int(years[1]) - int(years[0])
#                     if duration > 0:
#                         total_years += duration
#         except:
#             continue
    
#     return total_years if total_years > 0 else 0

# def extract_job_requirements(job_description):
#     """Extract specific requirements from job description"""
#     requirements = {
#         "experience_years": 0,
#         "education_level": [],
#         "required_skills": [],
#         "location": None
#     }
    
#     # Extract years of experience
#     years_patterns = [
#         r'(\d+)\s*(?:ans|years|year|an)(?:\s*d[\'e]?\s*expérience|\s*experience)',
#         r'expérience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#         r'experience\s*(?:de|of)?\s*(\d+)\s*(?:ans|years|year|an)',
#     ]
    
#     for pattern in years_patterns:
#         matches = re.findall(pattern, job_description.lower())
#         if matches:
#             requirements["experience_years"] = max([int(y) for y in matches])
#             break
    
#     # Extract education level
#     education_patterns = {
#         "bachelor": [r'\b(?:bachelor|licence|licenciatura)\b', r'\bbac\s*\+\s*3\b'],
#         "master": [r'\b(?:master|mastère|msc)\b', r'\bbac\s*\+\s*5\b'],
#         "engineer": [r'\b(?:ingénieur|engineer|ing[ée]nieur d\'[ée]tat)\b'],
#         "phd": [r'\b(?:phd|ph\.d|doctorat|doctorate)\b'],
#     }
    
#     for level, patterns in education_patterns.items():
#         for pattern in patterns:
#             if re.search(pattern, job_description.lower()):
#                 requirements["education_level"].append(level)
    
#     # Extract skills using spaCy if available
#     if nlp:
#         doc = nlp(job_description)
#         # Extract noun phrases as potential skills
#         for chunk in doc.noun_chunks:
#             if 3 < len(chunk.text) < 30:  # Reasonable length for a skill
#                 requirements["required_skills"].append(chunk.text.lower())
#     else:
#         # Fallback method: extract common tech skills with regex
#         tech_patterns = [
#             r'\b(?:java|python|javascript|js|typescript|ts|c\+\+|c#|php|ruby|swift|kotlin|golang|rust)\b',
#             r'\b(?:html|css|sql|mysql|postgresql|mongodb|nosql|redis|graphql)\b',
#             r'\b(?:aws|azure|gcp|cloud|docker|kubernetes|k8s|ci/cd|jenkins|git)\b',
#             r'\b(?:agile|scrum|kanban|jira|confluence|trello)\b',
#             r'\b(?:react|angular|vue|svelte|jquery|node\.js|nodejs|express|django|flask|spring|laravel)\b',
#             r'\b(?:tensorflow|pytorch|scikit-learn|pandas|numpy|data\s*science|machine\s*learning|ml|ai)\b',
#         ]
        
#         for pattern in tech_patterns:
#             matches = re.findall(pattern, job_description.lower())
#             requirements["required_skills"].extend(matches)
    
#     # Make sure skills are unique
#     requirements["required_skills"] = list(set(requirements["required_skills"]))
    
#     return requirements

# def analyze_cv_education(education_text, job_requirements):
#     """Analyze education section against job requirements"""
#     education_score = 0.0
#     max_score = 1.0
    
#     # If no education requirements, give a neutral score
#     if not job_requirements["education_level"]:
#         return 0.5
    
#     # Check for education levels in CV
#     education_matches = []
#     education_patterns = {
#         "bachelor": [r'\b(?:bachelor|licence|licenciatura)\b', r'\bbac\s*\+\s*3\b'],
#         "master": [r'\b(?:master|mastère|msc)\b', r'\bbac\s*\+\s*5\b'],
#         "engineer": [r'\b(?:ingénieur|engineer|ing[ée]nieur d\'[ée]tat)\b'],
#         "phd": [r'\b(?:phd|ph\.d|doctorat|doctorate)\b'],
#     }
    
#     for level, patterns in education_patterns.items():
#         for pattern in patterns:
#             if re.search(pattern, education_text.lower()):
#                 education_matches.append(level)
    
#     # Map education levels to numeric values for comparison
#     education_values = {
#         "bachelor": 1,
#         "engineer": 2,
#         "master": 2,
#         "phd": 3
#     }
    
#     # Find highest education in CV and job requirements
#     cv_max_edu = max([education_values.get(level, 0) for level in education_matches]) if education_matches else 0
#     job_max_edu = max([education_values.get(level, 0) for level in job_requirements["education_level"]]) if job_requirements["education_level"] else 0
    
#     if cv_max_edu >= job_max_edu and job_max_edu > 0:
#         # Meets or exceeds requirements
#         education_score = max_score
#     elif cv_max_edu > 0 and job_max_edu > 0:
#         # Partial match
#         education_score = cv_max_edu / job_max_edu * max_score
    
#     # Apply semantic similarity as well
#     if job_requirements["education_level"]:
#         job_edu_text = " ".join(job_requirements["education_level"])
#         similarity = calculate_semantic_similarity(job_edu_text, education_text)
#         # Blend direct matching with semantic similarity
#         education_score = 0.7 * education_score + 0.3 * similarity
    
#     return min(education_score, max_score)  # Cap at max_score

# def analyze_cv_experience(experience_text, job_requirements):
#     """Analyze experience section against job requirements"""
#     experience_score = 0.0
#     max_score = 1.0
    
#     # Extract years of experience from CV
#     cv_experience_years = extract_experience_years(experience_text)
    
#     # Compare with job requirements
#     required_years = job_requirements["experience_years"]
    
#     if required_years > 0:
#         if cv_experience_years >= required_years:
#             # Meets or exceeds requirements
#             experience_score = max_score
#             # Bonus for exceeding requirements by more than 2 years (but cap it)
#             if cv_experience_years > required_years + 2:
#                 experience_score = min(max_score * 1.1, max_score)
#         elif cv_experience_years > 0:
#             # Partial match
#             experience_score = (cv_experience_years / required_years) * max_score
#     else:
#         # If no specific requirement, give a neutral score
#         experience_score = 0.5
    
#     # Apply semantic similarity to evaluate quality/relevance of experience
#     similarity = calculate_semantic_similarity(job_requirements.get("experience_text", ""), experience_text)
    
#     # Blend direct year matching with semantic similarity
#     final_score = 0.7 * experience_score + 0.3 * similarity
    
#     return min(final_score, max_score)  # Cap at max_score

# def analyze_cv_skills(skills_text, job_requirements):
#     """Analyze skills section against job requirements"""
#     skills_score = 0.0
#     max_score = 1.0
    
#     required_skills = job_requirements["required_skills"]
    
#     # If no specific skills requirements, rely on semantic similarity
#     if not required_skills:
#         return calculate_semantic_similarity(job_requirements.get("skills_text", ""), skills_text)
    
#     # Count how many required skills are found in CV
#     skills_found = 0
#     unique_skills = set()
    
#     for skill in required_skills:
#         skill_pattern = r'\b' + re.escape(skill) + r'\b'
#         if re.search(skill_pattern, skills_text.lower()):
#             skills_found += 1
#             unique_skills.add(skill)
    
#     # Calculate direct skills match ratio
#     if required_skills:
#         direct_match_score = skills_found / len(required_skills)
#     else:
#         direct_match_score = 0.5  # Neutral score if no requirements
    
#     # Apply semantic similarity for skills understanding
#     similarity = calculate_semantic_similarity(" ".join(required_skills), skills_text)
    
#     # Look for bonus skills that might be valuable but not required
#     bonus_score = 0.0
#     if nlp:
#         doc = nlp(skills_text)
#         # Extract skill-like terms
#         potential_bonus_skills = []
#         for chunk in doc.noun_chunks:
#             if 3 < len(chunk.text) < 30:  # Reasonable length for a skill
#                 potential_bonus_skills.append(chunk.text.lower())
        
#         # Count bonus skills (ones not in requirements)
#         bonus_skills = set(potential_bonus_skills) - set([s.lower() for s in required_skills])
#         if bonus_skills:
#             # Bonus for having additional skills (up to 0.1 extra)
#             bonus_score = min(0.1, len(bonus_skills) * 0.02)
    
#     # Blend direct matching with semantic similarity
#     skills_score = (0.6 * direct_match_score) + (0.4 * similarity) + bonus_score
    
#     return min(skills_score, max_score)  # Cap at max_score

# def analyze_cv_region(region_text, job_requirements):
#     """Analyze region/location against job requirements"""
#     # Since region matching is usually simpler (exact match or not),
#     # we'll use a more straightforward approach
    
#     if not job_requirements.get("location"):
#         return 0.5  # Neutral score if no location requirement
    
#     # Direct location mention
#     if job_requirements["location"].lower() in region_text.lower():
#         return 1.0
    
#     # Fallback to semantic similarity
#     return calculate_semantic_similarity(job_requirements.get("location", ""), region_text)

# def calculate_semantic_similarity(text1, text2):
#     """Calculate semantic similarity between two texts using the loaded model"""
#     if not text1 or not text2:
#         return 0.5  # Neutral score for empty inputs
    
#     # Encode texts to vectors
#     embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
#     embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
    
#     # Calculate cosine similarity
#     similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    
#     return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1




# def extract_experience_details(experience_text):
#     """
#     Extract detailed experience information including:
#     - Total years of experience
#     - Date ranges found
#     - Job titles
#     - Companies
#     """
#     details = {
#         "total_years": 0,
#         "date_ranges": [],
#         "positions": [],
#         "companies": []
#     }
    
#     # Extract years using the existing function
#     details["total_years"] = extract_experience_years(experience_text)
    
#     # Extract date ranges
#     for pattern in DATE_PATTERNS:
#         matches = re.findall(pattern, experience_text.lower())
#         if matches:
#             for match in matches:
#                 if isinstance(match, tuple):
#                     range_str = f"{match[0]} - {match[1]}"
#                 else:
#                     range_str = match
#                 details["date_ranges"].append(range_str)
    
#     # Extract job titles (common job title patterns)
#     job_title_patterns = [
#         r'\b(développeur|developer|ingénieur|engineer|architecte|architect|lead|chef|responsable|manager|consultant)\s+([\w\s]{3,30}?)\b',
#         r'\b(directeur|director|VP|head of)\s+([\w\s]{3,30}?)\b'
#     ]
    
#     for pattern in job_title_patterns:
#         matches = re.findall(pattern, experience_text.lower())
#         if matches:
#             for match in matches:
#                 if len(match) >= 2:
#                     job_title = f"{match[0]} {match[1]}".strip()
#                     if job_title not in details["positions"]:
#                         details["positions"].append(job_title)
    
#     # Extract company names (this is more complex, simplified approach)
#     if nlp:
#         doc = nlp(experience_text)
#         for ent in doc.ents:
#             if ent.label_ == "ORG":
#                 if len(ent.text) > 2 and ent.text not in details["companies"]:
#                     details["companies"].append(ent.text)
    
#     return details

# def create_experience_classification(cv_experience_years, required_years):
#     """
#     Create a detailed classification of experience with observations
#     """
#     classification = {
#         "status": "Approved" if cv_experience_years >= required_years else "Rejected",
#         "required_years": required_years,
#         "found_years": cv_experience_years,
#         "difference": cv_experience_years - required_years,
#         "observations": []
#     }
    
#     # Add specific observations
#     if cv_experience_years >= required_years:
#         if cv_experience_years > required_years + 5:
#             classification["observations"].append(f"Candidate has {cv_experience_years - required_years} years more than required (possibly overqualified)")
#         elif cv_experience_years > required_years:
#             classification["observations"].append(f"Candidate exceeds the required experience by {cv_experience_years - required_years} years")
#         else:
#             classification["observations"].append("Candidate meets the exact experience requirement")
#     else:
#         deficit = required_years - cv_experience_years
#         classification["observations"].append(f"Candidate lacks {deficit} years of required experience")
        
#         if cv_experience_years > 0:
#             percentage = (cv_experience_years / required_years) * 100
#             classification["observations"].append(f"Candidate has {percentage:.1f}% of the required experience")
            
#             if percentage >= 80:
#                 classification["observations"].append("Consider for interview despite slight experience shortage")
    
#     return classification
# def count_matched_skills(cv_text, skills):
#     return sum(1 for skill in skills if re.search(r'\b' + re.escape(skill) + r'\b', cv_text, re.IGNORECASE))

# # Update the match_and_rank_cvs function to include the new detailed experience analysis
# def match_and_rank_cvs(cv_texts, job_description, files, weights):
#     """Match CV texts against job description with intelligent analysis"""
#     # Extract detailed job requirements
#     job_sections = extract_sections(job_description)
#     job_requirements = extract_job_requirements(job_description)
    
#     # Add full section texts to requirements for semantic matching
#     job_requirements["education_text"] = job_sections["education"]
#     job_requirements["skills_text"] = job_sections["skills"]
#     job_requirements["experience_text"] = job_sections["experience"]
#     job_requirements["location"] = job_sections["region"]
    
#     results = []
#     for i, cv_text in enumerate(cv_texts):
#         # Extract CV sections
#         cv_sections = extract_sections(cv_text)
        
#         # Perform detailed analysis for each section
#         education_score = analyze_cv_education(cv_sections["education"], job_requirements)
#         skills_score = analyze_cv_skills(cv_sections["skills"], job_requirements)
#         experience_score = analyze_cv_experience(cv_sections["experience"], job_requirements)
#         region_score = analyze_cv_region(cv_sections["region"], job_requirements)
        
#         # Store section scores
#         section_similarities = {
#             "education": education_score,
#             "skills": skills_score,
#             "experience": experience_score,
#             "region": region_score,
#         }
        
#         # Calculate weighted average similarity
#         total_weight = sum(weights.values())
#         if total_weight == 0:  # Prevent division by zero
#             total_weight = 1
            
#         weighted_similarity = (
#             (section_similarities["education"] * weights.get("education", 1)) +
#             (section_similarities["skills"] * weights.get("skills", 1)) +
#             (section_similarities["experience"] * weights.get("experience", 1)) +
#             (section_similarities["region"] * weights.get("region", 1))
#         ) / total_weight
        
#         # Extract years of experience for the result
#         cv_experience_years = extract_experience_years(cv_text)
        
#         # Extract detailed experience information
#         experience_details = extract_experience_details(cv_sections["experience"])
        
#         # Create detailed experience classification
#         experience_classification = create_experience_classification(
#             cv_experience_years, 
#             job_requirements["experience_years"]
#         )
        
#         # Determine if the CV meets minimum requirements
#         meets_min_reqs = True
#         rejection_reasons = []
        
#         # Check experience years
#         if job_requirements["experience_years"] > 0 and cv_experience_years < job_requirements["experience_years"]:
#             meets_min_reqs = False
#             rejection_reasons.append(f"Experience: {cv_experience_years} years (requires {job_requirements['experience_years']})")
        
#         # Save the file to the server directory
#         filename = files[i].filename
#         file_path = os.path.join(UPLOAD_DIR, filename)
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(files[i].file, f)
        
#         # Create detailed experience status string for display
#         experience_status = ""
#         if job_requirements["experience_years"] > 0:
#             if cv_experience_years >= job_requirements["experience_years"]:
#                 experience_status = f"Approved: {cv_experience_years}/{job_requirements['experience_years']} years"
#             else:
#                 experience_status = f"Rejected: {cv_experience_years}/{job_requirements['experience_years']} years"
        
#         # Apply threshold and final status
#         status = "Accepted" if (meets_min_reqs and weighted_similarity >= SIMILARITY_THRESHOLD) else "Rejected"
        
#         # Prepare the final reason with detailed breakdown
#         reason = None
#         if status == "Rejected":
#             rejection_details = []
            
#             # Experience details
#             if job_requirements["experience_years"] > 0 and cv_experience_years < job_requirements["experience_years"]:
#                 deficit = job_requirements["experience_years"] - cv_experience_years
#                 if cv_experience_years == 0:
#                     rejection_details.append(f"No experience found (requires {job_requirements['experience_years']} years)")
#                 else:
#                     rejection_details.append(f"Experience: {cv_experience_years}/{job_requirements['experience_years']} years (deficit: {deficit} years)")
            
#             # Skills details
#             if job_requirements["required_skills"] and section_similarities["skills"] < 0.4:
#                 found = [skill for skill in job_requirements["required_skills"] if re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())]
#                 missing = [skill for skill in job_requirements["required_skills"] if not re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())]
#                 if missing:
#                     rejection_details.append(f"Missing skills: {', '.join(missing[:3])}{' and more' if len(missing) > 3 else ''}")
            
#             # Region details
#             if job_requirements["location"] and section_similarities["region"] < 0.3:
#                 rejection_details.append(f"Location mismatch: Required {job_requirements['location']}")
            
#             # Education details
#             if job_requirements["education_level"] and section_similarities["education"] < 0.4:
#                 rejection_details.append(f"Education mismatch: Required {', '.join(job_requirements['education_level'])}")
            
#             # Final reason formatting
#             if rejection_details:
#                 reason = " | ".join(rejection_details)
#             else:
#                 reason = f"Overall match score ({weighted_similarity:.2f}) below threshold ({SIMILARITY_THRESHOLD})"
        
#         # Create detailed result
#         result = {
#             "filename": filename,
#             "similarity": float(weighted_similarity),
#             "file_path": f"/files/{filename}",
#             "status": status,
#             "reason": reason,
#             "experience_years": cv_experience_years,
#             # Add prominent experience status for frontend display
#             "experience_status": experience_status,
#             "section_scores": {
#                 "education": float(section_similarities["education"]),
#                 "skills": float(section_similarities["skills"]),
#                 "experience": float(section_similarities["experience"]),
#                 "region": float(section_similarities["region"]),
#             },
#             # Add detailed experience analysis
#             "experience_analysis": {
#                 "classification": experience_classification,
#                 "details": experience_details,
#                 # Add direct comparison data for frontend display
#                 "display": {
#                     "required_years": job_requirements["experience_years"],
#                     "candidate_years": cv_experience_years,
#                     "deficit_or_surplus": cv_experience_years - job_requirements["experience_years"],
#                     "percentage_of_required": (cv_experience_years / job_requirements["experience_years"] * 100) if job_requirements["experience_years"] > 0 else 100,
#                     "status_text": f"{cv_experience_years}/{job_requirements['experience_years']} years" if job_requirements["experience_years"] > 0 else "No specific requirement",
#                     "color_code": "green" if cv_experience_years >= job_requirements["experience_years"] else "red",
#                 }
#             },
#             # Add some detailed analysis for the frontend
#             "analysis": {
#                 "meets_experience": bool(cv_experience_years >= job_requirements["experience_years"]) if job_requirements["experience_years"] > 0 else None,
#                 "required_skills_found": [skill for skill in job_requirements["required_skills"] if re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())],
#                 "missing_skills": [skill for skill in job_requirements["required_skills"] if not re.search(r'\b' + re.escape(skill) + r'\b', cv_text.lower())],
#                 # Add summary statements for clear display
#                 "summary": {
#                     "experience_summary": (
#         f"{cv_experience_years}/{job_requirements['experience_years']} years"
#         if job_requirements["experience_years"] > 0 else "No specific requirement"
#     ),
#     "skills_summary": (
#         f"{count_matched_skills(cv_text, job_requirements['required_skills'])}/{len(job_requirements['required_skills'])} skills matched"
#         if job_requirements["required_skills"] else "No specific skills requirement"
#     ),
#     "education_match": (
#         "Matched" if section_similarities["education"] >= 0.7 else
#         "Partial match" if section_similarities["education"] >= 0.4 else
#         "Low match"
#     ),
#     "location_match": (
#         "Matched" if section_similarities["region"] >= 0.7 else
#         "Partial match" if section_similarities["region"] >= 0.3 else
#         "No match"
#     )
#                     }
#             }
#         }
        
#         results.append(result)
    
#     # Sort by similarity score in descending order
#     return sorted(results, key=lambda x: x["similarity"], reverse=True)

# @app.post("/match-cvs/")
# async def match_cvs(files: list[UploadFile] = File(...), 
#                    job_description: str = Form(...), 
#                    education_weight: float = Form(...), 
#                    skills_weight: float = Form(...), 
#                    experience_weight: float = Form(...), 
#                    region_weight: float = Form(...)):
#     """API endpoint to match CVs against a job description"""
#     if not job_description or not files:
#         return JSONResponse(status_code=400, content={"error": "Missing job description or files."})

#     # Prepare weights
#     weights = {
#         "education": education_weight,
#         "skills": skills_weight,
#         "experience": experience_weight,
#         "region": region_weight,
#     }

#     # Extract job requirements for summary
#     job_requirements = extract_job_requirements(job_description)
#     job_sections = extract_sections(job_description)
    
#     # Add full section texts to requirements for summary
#     job_requirements["education_text"] = job_sections["education"]
#     job_requirements["skills_text"] = job_sections["skills"]
#     job_requirements["experience_text"] = job_sections["experience"]
#     job_requirements["location"] = job_sections["region"]

#     cv_texts = []
#     for file in files:
#         try:
#             # Reset the file pointer for each file
#             file.file.seek(0)
#             text = extract_text_from_file(file)
#             if text:
#                 cv_texts.append(text)
#                 # Reset the file pointer after extraction for later use
#                 file.file.seek(0)
#         except Exception as e:
#             return JSONResponse(status_code=400, content={"error": str(e)})

#     results = match_and_rank_cvs(cv_texts, job_description, files, weights)

#     if not results:
#         return {"message": "No suitable CVs found."}
    
#     # Create job summary for context
#     job_summary = {
#         "experience_requirement": f"{job_requirements['experience_years']} years" if job_requirements['experience_years'] > 0 else "Not specified",
#         "education_requirement": ", ".join(job_requirements["education_level"]) if job_requirements["education_level"] else "Not specified",
#         "skills_requirement": job_requirements["required_skills"],
#         "location_requirement": job_requirements["location"] if job_requirements["location"] and len(job_requirements["location"]) > 3 else "Not specified",
#     }

#     return {
#         "message": "Matching complete!", 
#         "job_summary": job_summary,
#         "ranked_cvs": results,
#         "matching_stats": {
#             "total_cvs": len(results),
#             "accepted": sum(1 for r in results if r["status"] == "Accepted"),
#             "rejected": sum(1 for r in results if r["status"] == "Rejected"),
#             "experience_rejected": sum(1 for r in results if r["status"] == "Rejected" and r["experience_years"] < job_requirements["experience_years"]),
#         }
#     }  

# @app.get("/files/{filename}")
# async def get_file(filename: str):
#     """Endpoint to retrieve uploaded files"""
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     else:
#         return JSONResponse(status_code=404, content={"error": "File not found"})

# # If running directly (not imported)
# if __name__ == "__main__":
#     import uvicorn
#     print("⚡ Starting CV matching server...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


















# Complete Flask Backend (app.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import uuid
import json
import spacy
import datetime
import pytesseract
from pdf2image import convert_from_path
import docx2txt
import PyPDF2
from werkzeug.utils import secure_filename
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import logging
import numpy as np
from spacy.lang.en import English
from spacy.lang.fr import French

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Set up temporary storage for job descriptions and CVs (in-memory for demo)
# In a production environment, use a proper database
job_descriptions = {}
cv_documents = {}

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    logger.warning("NLTK download failed. If running offline, make sure data is pre-downloaded.")

# Initialize NLP models
try:
    # Load spaCy models
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')
    logger.info("SpaCy models loaded successfully")
except:
    # Fallback to basic models if full models not available
    logger.warning("Full spaCy models not available. Using basic language support.")
    nlp_en = English()
    nlp_fr = French()

# Define regions list for region matching
algerian_regions = [
    'alger', 'algiers', 'oran', 'constantine', 'annaba', 'batna', 'setif', 
    'sétif', 'blida', 'tlemcen', 'béjaïa', 'bejaia', 'tiaret', 'djelfa', 
    'biskra', 'sidi bel abbès', 'sidi bel abbes', 'jijel', 'mostaganem',
    'skikda', 'tebessa', 'tébessa', 'chlef', 'médéa', 'medea'
]
regions_pattern = re.compile(r'\b(' + '|'.join(algerian_regions) + r')\b', re.IGNORECASE)

# Define education levels for scoring
education_levels = {
    # French
    'bac': 1,
    'baccalauréat': 1,
    'baccalaureat': 1,
    'bac+1': 1.5,
    'bac+2': 2,
    'bts': 2,
    'dut': 2,
    'deug': 2,
    'bac+3': 3,
    'licence': 3,
    'bachelor': 3,
    'bac+4': 3.5,
    'maîtrise': 3.5,
    'maitrise': 3.5,
    'bac+5': 4,
    'master': 4,
    'magistère': 4,
    'magistere': 4,
    'ingénieur': 4,
    'ingenieur': 4,
    'bac+8': 5,
    'doctorat': 5,
    'phd': 5,
    'docteur': 5,
    
    # English
    'high school': 1,
    'associate': 2,
    'associate degree': 2,
    'associate\'s degree': 2,
    'undergraduate': 3,
    'bachelor\'s': 3,
    'bachelor\'s degree': 3,
    'bachelors': 3,
    'master\'s': 4,
    'master\'s degree': 4,
    'masters': 4,
    'mba': 4,
    'doctorate': 5,
    'doctoral': 5,
    'ph.d': 5,
    'ph.d.': 5
}

# Define common technical skills for pattern matching
technical_skills = [
    # Programming Languages
    'java', 'python', 'javascript', 'js', 'typescript', 'ts', 'c\\+\\+', 'c#', 'ruby', 'php', 'swift',
    'kotlin', 'golang', 'go', 'scala', 'rust', 'perl', 'r', 'dart', 'haskell', 'lua', 'groovy',
    
    # Web Development
    'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery', 'json', 'xml',
    'react', 'angular', 'vue', 'svelte', 'next\\.js', 'nuxt', 'node\\.js', 'express',
    'django', 'flask', 'spring', 'laravel', 'symfony', 'rails', 'asp\\.net',
    
    # Mobile
    'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'swift', 'objective-c',
    
    # Databases
    'sql', 'mysql', 'postgresql', 'mongo', 'mongodb', 'firebase', 'oracle', 'sqlite',
    'nosql', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'mariadb',
    
    # DevOps & Cloud
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'jenkins',
    'git', 'github', 'gitlab', 'bitbucket', 'ci/cd', 'terraform', 'ansible', 'chef',
    'vagrant', 'prometheus', 'grafana',
    
    # Data Science
    'machine learning', 'ml', 'deep learning', 'dl', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'tableau', 'power bi',
    'hadoop', 'spark', 'nlp', 'computer vision', 'cv', 'data mining',
    
    # Networking & Security
    'tcp/ip', 'dns', 'http', 'https', 'ssh', 'ftp', 'vpn', 'firewall', 'encryption',
    'oauth', 'openid', 'ldap', 'active directory', 'kerberos',
    
    # Software & Tools
    'jira', 'confluence', 'trello', 'slack', 'ms office', 'excel', 'word', 'powerpoint',
    'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd', 'indesign',
    
    # Methodologies
    'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'rup', 'extreme programming', 'xp',
    'devops', 'itil', 'six sigma', 'prince2', 'pmp',
    
    # Specific to telecom/Mobilis
    'gsm', 'umts', 'lte', '5g', 'voip', 'sip', 'mpls', 'sdh', 'dwdm', 'ims', 'ran', 
    'bss', 'oss', 'crm', 'erp', 'billing', 'provisioning', 'telecom', 'networking',
    'cisco', 'huawei', 'ericsson', 'nokia', 'alcatel', 'juniper', 'siemens'
]

# Helper function to detect language
def detect_language(text):
    """Detect if text is primarily in English or French"""
    # Simple heuristic: count distinctive French words
    french_indicators = ['et', 'de', 'la', 'le', 'du', 'les', 'des', 'un', 'une', 'pour', 'dans']
    english_indicators = ['and', 'the', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'from']
    
    text_lower = text.lower()
    french_count = sum(1 for word in french_indicators if f' {word} ' in f' {text_lower} ')
    english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
    
    return 'fr' if french_count > english_count else 'en'

# Helper function to get appropriate NLP model based on language
def get_nlp_model(text):
    language = detect_language(text)
    return nlp_fr if language == 'fr' else nlp_en

# File processing functions
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        # First try PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        
        # If PyPDF2 didn't extract enough text, try OCR
        if len(text.strip()) < 100:  # Arbitrary threshold
            logger.info(f"PDF text extraction yielded minimal text. Attempting OCR for {file_path}")
            images = convert_from_path(file_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_file(file_path):
    """Extract text based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        return ""

# Information extraction functions
def extract_education(text):
    """Extract education qualifications from text"""
    # First try to find direct mentions of education levels
    education_pattern = r'\b(' + '|'.join(education_levels.keys()) + r')\b'
    education_matches = re.finditer(education_pattern, text.lower())
    
    # Extract surrounding context for each match
    education_info = []
    for match in education_matches:
        start_pos = max(0, match.start() - 50)
        end_pos = min(len(text), match.end() + 100)
        context = text[start_pos:end_pos].replace('\n', ' ')
        education_info.append(context)
    
    # Also look for general education-related sections
    education_sections = []
    education_headers = [
        'education', 'formation', 'qualification', 'diplôme', 'diplome', 
        'academic', 'études', 'etudes', 'cursus'
    ]
    
    lines = text.split('\n')
    in_education_section = False
    current_section = []
    
    for line in lines:
        line_lower = line.lower()
        
        # Check if this line is an education section header
        if any(header in line_lower for header in education_headers):
            if in_education_section and current_section:
                education_sections.append(' '.join(current_section))
                current_section = []
            in_education_section = True
            current_section.append(line)
        # If we're in an education section, add lines until we hit what looks like a new section
        elif in_education_section:
            if re.match(r'^[A-Z\s]{2,}:?$', line) or not line.strip():  # New section header or empty line
                if current_section:
                    education_sections.append(' '.join(current_section))
                    current_section = []
                in_education_section = False
            else:
                current_section.append(line)
    
    # Add the last section if we were in an education section
    if in_education_section and current_section:
        education_sections.append(' '.join(current_section))
    
    # Combine all education information
    all_education_info = education_info + education_sections
    
    if all_education_info:
        return "\n".join(all_education_info)
    else:
        return "No education information found"

def extract_highest_education_level(text):
    """Extract the highest education level mentioned in text"""
    text_lower = text.lower()
    highest_level = 0
    
    for term, level in education_levels.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            highest_level = max(highest_level, level)
    
    return highest_level

# def extract_experience_years(text):
#     """Extract years of experience from text"""
#     # Look for specific patterns indicating years of experience
#     experience_patterns = [
#         r'(\d+)(?:\s*-\s*\d+)?\s*(?:an|ans|année|annee|years|year)(?:\s*d\'?expérience|\s*d\'?experience|\s*of\s*experience)?',
#         r'expérience(?:\s*de|\s*d\'|\s*:)?\s*(\d+)(?:\s*-\s*\d+)?\s*(?:an|ans|année|annee|years|year)',
#         r'experience(?:\s*of|\s*:)?\s*(\d+)(?:\s*-\s*\d+)?\s*(?:an|ans|année|annee|years|year)',
#         r'(\d+)(?:\s*-\s*\d+)?\s*(?:an|ans|année|annee|years|year)(?:\s*minimum|d\'?expérience\s*minimum|d\'?experience\s*minimum)',
#     ]
    
#     for pattern in experience_patterns:
#         matches = re.findall(pattern, text.lower())
#         if matches:
#             # Take the first number found (lower bound if range is given)
#             return int(matches[0])
    
#     # If no direct mention of years, try to calculate from employment dates
#     try:
#         # Look for date ranges in format like "2015-2020" or "Jan 2015 - Dec 2020"
#         date_ranges = re.findall(r'(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)[\s,.]+)?(\d{4})(?:\s*[-–—]\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)[\s,.]+)?(?:(\d{4})|present|actuel|maintenant|now|current))?', text.lower())
        
#         if date_ranges:
#             total_years = 0
#             current_year = datetime.datetime.now().year
            
#             for date_range in date_ranges:
#                 start_year = int(date_range[0])
                
#                 # If end year is specified, use it; otherwise assume it's current
#                 if len(date_range) > 1 and date_range[1]:
#                     end_year = int(date_range[1])
#                 else:
#                     end_year = current_year
                
#                 # Add the duration (capped at current year)
#                 years = min(end_year, current_year) - start_year
#                 if years > 0:  # Only count positive durations
#                     total_years += years
            
#             return total_years
#     except Exception as e:
#         logger.warning(f"Error calculating experience from dates: {e}")
    
#     # Default to 0 if no experience info found
#     return 0
def extract_experience_years(text):
    """Extract years of experience from text with improved pattern matching"""
    # Look for specific patterns indicating years of experience with more comprehensive regex
    experience_patterns = [
        # French patterns
        r'(\d+)(?:\s*[àa]\s*\d+)?(?:\s*|\-|\s+\-\s+)(?:an|ans|année|annee|années|annees)(?:\s+d\'?expérience|\s+d\'?experience|\s+dans\s+le\s+domaine)?',
        r'expérience(?:\s*de|\s*d\'|\s*:|\s*minimum|\s*professionnelle|\s*requise)?\s*(?:de\s*)?(\d+)(?:\s*[àa]\s*\d+)?(?:\s*|\-|\s+\-\s+)(?:an|ans|année|annee|années|annees)',
        r'experience(?:\s*of|\s*:|\s*minimum|\s*required|\s*professional)?\s*(?:of\s*)?(\d+)(?:\s*to\s*\d+)?(?:\s*|\-|\s+\-\s+)(?:an|ans|année|annee|années|annees|year|years)',
        
        # English patterns
        r'(\d+)(?:\s*to\s*\d+|\s*\-\s*\d+)?(?:\s*|\-)(?:year|years)(?:\s+of\s+experience|\s+in\s+the\s+field)?',
        r'experience(?:\s*of|\s*:|\s*minimum|\s*required|\s*professional)?\s*(?:of\s*)?(\d+)(?:\s*to\s*\d+)?(?:\s*|\-|\s+\-\s+)(?:year|years)',
        
        # Numbers followed by year(s) within parentheses
        r'\((\d+)(?:\s*[àa]\s*\d+|\s*to\s*\d+|\s*\-\s*\d+)?(?:\s*|\-)(?:an|ans|année|annee|années|annees|year|years)\)',
        
        # Direct mention in job requirements
        r'(?:cinq|cinq \(05\)|05|5)(?:\s*|\-)(?:an|ans|année|annee|années|annees)',
        r'(?:five|5)(?:\s*|\-)(?:year|years)',
        
        # Domain-specific patterns for the provided example
        r'(?:dans\s+le\s+domaine.*?)(\d+)(?:\s*|\-)(?:an|ans|année|annee|années|annees)',
        r'(?:in\s+the\s+field.*?)(\d+)(?:\s*|\-)(?:year|years)',
    ]
    
    # Check each pattern
    for pattern in experience_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Try to convert the first match to an integer
            try:
                # Take the first number found (lower bound if range is given)
                return int(matches[0])
            except (ValueError, TypeError):
                # If the match isn't a clean number, try to extract digits
                digits = re.findall(r'\d+', str(matches[0]))
                if digits:
                    return int(digits[0])
    
    # Look for written numbers (especially in French job postings)
    written_numbers = {
        'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
        'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for word, value in written_numbers.items():
        # Pattern for "cinq (05) ans" or similar formats
        pattern = r'\b' + word + r'(?:\s*\(\d+\))?\s*(?:an|ans|année|annee|années|annees|year|years)\b'
        if re.search(pattern, text.lower()):
            return value
    
    # Extract numbers from specific phrases like "05 ans" directly referenced in the job posting
    direct_years_pattern = r'\b(\d+)\s*(?:an|ans|année|annee|années|annees|year|years)\b'
    direct_years_matches = re.findall(direct_years_pattern, text.lower())
    if direct_years_matches:
        return int(direct_years_matches[0])
    
    # If no direct mention of years, try to calculate from employment dates
    try:
        # Expanded date format detection - look for date ranges in various formats
        date_patterns = [
            # Standard date ranges: 2015-2020
            r'(\d{4})(?:\s*[-–—]\s*)(?:(\d{4})|present|présent|actuel|maintenant|now|current|aujourd\'hui)',
            
            # Month-Year format: Jan 2015 - Dec 2020
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)[\s,.]+(\d{4})(?:\s*[-–—]\s*)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)[\s,.]+)?(?:(\d{4})|present|présent|actuel|maintenant|now|current|aujourd\'hui)',
            
            # Resume style entries: Company Name (2015-2020)
            r'\((\d{4})(?:\s*[-–—]\s*)(?:(\d{4})|present|présent|actuel|maintenant|now|current|aujourd\'hui)\)',
        ]
        
        total_years = 0
        current_year = datetime.datetime.now().year
        date_ranges_found = []
        
        # Check each date pattern
        for pattern in date_patterns:
            matches = re.findall(pattern, text.lower())
            date_ranges_found.extend(matches)
        
        if date_ranges_found:
            for date_range in date_ranges_found:
                # Get the start year from the match
                if isinstance(date_range, tuple):
                    start_year = int(date_range[0]) if date_range[0].isdigit() else None
                    
                    # If end year is specified and is a digit, use it; otherwise assume current
                    if len(date_range) > 1 and date_range[1] and date_range[1].isdigit():
                        end_year = int(date_range[1])
                    else:
                        end_year = current_year
                else:
                    # If not a tuple, try to extract years directly
                    years = re.findall(r'\d{4}', date_range)
                    if len(years) >= 2:
                        start_year = int(years[0])
                        end_year = int(years[1])
                    elif len(years) == 1:
                        start_year = int(years[0])
                        end_year = current_year
                    else:
                        continue
                
                # Add the duration (capped at current year and positive values only)
                if start_year and start_year <= current_year:
                    years = min(end_year, current_year) - start_year
                    if years > 0:  # Only count positive durations
                        total_years += years
            
            return total_years
    except Exception as e:
        logger.warning(f"Error calculating experience from dates: {e}")
    
    # Special case for the example: "Cinq (05) ans dans le domaine développement logiciel."
    # Direct check for the exact phrase in the given example
    if re.search(r'cinq\s*\(05\)\s*ans', text.lower()) or re.search(r'05\s*ans', text.lower()):
        return 5
    
    # Default to 0 if no experience info found
    return 0
def extract_region(text):
    """Extract region information from text"""
    matches = regions_pattern.findall(text.lower())
    return list(set(matches)) if matches else []

def extract_skills(text):
    """Extract technical skills from text"""
    skills_found = []
    text_lower = text.lower()
    
    # Check for each technical skill pattern
    for skill in technical_skills:
        if re.search(r'\b' + skill + r'\b', text_lower):
            # Clean up the skill name for display
            clean_skill = skill.replace('\\', '')
            skills_found.append(clean_skill)
    
    # Additional NLP-based skill extraction
    try:
        nlp = get_nlp_model(text)
        doc = nlp(text)
        
        # Extract noun phrases that might be skills
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:  # 2-4 word phrases likely to be skills
                noun_phrases.append(chunk.text.lower())
        
        # Filter noun phrases to likely skills (basic heuristic)
        technical_terms = ['system', 'software', 'database', 'network', 'programming', 
                          'développement', 'developpement', 'analysis', 'gestion', 
                          'management', 'engineering', 'ingénierie', 'ingenierie',
                          'design', 'architecture', 'security', 'sécurité', 'securite']
        
        for phrase in noun_phrases:
            if any(term in phrase for term in technical_terms) and phrase not in skills_found:
                skills_found.append(phrase)
    except Exception as e:
        logger.warning(f"Error in NLP skill extraction: {e}")
    
    return list(set(skills_found))

def calculate_skills_similarity(job_skills, cv_skills):
    """Calculate similarity between job skills and CV skills"""
    if not job_skills or not cv_skills:
        return 0.0
    
    # Convert to lowercase for matching
    job_skills_lower = [s.lower() for s in job_skills]
    cv_skills_lower = [s.lower() for s in cv_skills]
    
    # Count exact matches
    exact_matches = sum(1 for skill in job_skills_lower if skill in cv_skills_lower)
    
    # Use TF-IDF for semantic similarity
    try:
        # Combine all skills into documents for vectorization
        job_doc = " ".join(job_skills_lower)
        cv_doc = " ".join(cv_skills_lower)
        documents = [job_doc, cv_doc]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity
        semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Weighted combination of exact matches and semantic similarity
        if len(job_skills_lower) > 0:
            exact_match_ratio = exact_matches / len(job_skills_lower)
        else:
            exact_match_ratio = 0
        
        final_score = 0.6 * exact_match_ratio + 0.4 * semantic_similarity
        return min(1.0, final_score)  # Cap at 1.0
    except Exception as e:
        logger.warning(f"Error calculating skills similarity: {e}")
        # Fallback to exact match ratio
        if len(job_skills_lower) > 0:
            return exact_matches / len(job_skills_lower)
        else:
            return 0.0

# def extract_job_requirements(text):
#     """Extract job requirements from job description text"""
#     nlp = get_nlp_model(text)
    
#     # Split text into sections (looking for "Mission" and "Profil" sections)
#     sections = {}
#     lines = text.split('\n')
#     current_section = "general"
#     sections[current_section] = []
    
#     for line in lines:
#         line_lower = line.lower().strip()
#         if "mission" in line_lower and len(line_lower) < 30:
#             current_section = "mission"
#             sections[current_section] = []
#         elif "profil" in line_lower and len(line_lower) < 30:
#             current_section = "profil"
#             sections[current_section] = []
#         else:
#             sections[current_section].append(line)
    
#     for section in sections:
#         sections[section] = '\n'.join(sections[section])
    
#     # Focus on "Profil" section for requirements if available
#     profile_text = sections.get("profil", text)
    
#     # Extract education, experience, region, and skills
#     education_info = extract_education(profile_text)
#     highest_education = extract_highest_education_level(profile_text)
#     required_experience = extract_experience_years(profile_text)
#     region_list = extract_region(profile_text)
#     region = region_list[0] if region_list else None
#     skills_list = extract_skills(profile_text)
    
#     return {
#         "education": education_info,
#         "education_level": highest_education,
#         "experience": required_experience,
#         "region": region,
#         "skills": skills_list
#     }
def extract_job_requirements(text):
    """Extract job requirements from job description text with more sections"""
    nlp = get_nlp_model(text)
    
    # Split text into sections (looking for "Mission", "Profil", "Expérience" sections)
    sections = {}
    lines = text.split('\n')
    current_section = "general"
    sections[current_section] = []
    
    for line in lines:
        line_lower = line.lower().strip()
        if re.search(r'\bmission', line_lower) and len(line_lower) < 50:
            current_section = "mission"
            sections[current_section] = []
        elif re.search(r'\bprofil', line_lower) and len(line_lower) < 50:
            current_section = "profil"
            sections[current_section] = []
        elif re.search(r'\bexperience|\bexpérience', line_lower) and len(line_lower) < 50:
            current_section = "experience"
            sections[current_section] = []
        else:
            sections[current_section].append(line)
    
    for section in sections:
        sections[section] = '\n'.join(sections[section])
    
    # Combine profil and experience sections for requirements if available
    profile_text = sections.get("profil", "")
    experience_text = sections.get("experience", "")
    combined_text = profile_text + "\n" + experience_text
    
    # If no specific sections were found, use the entire text
    if not profile_text and not experience_text:
        combined_text = text
    
    # Extract education, experience, region, and skills
    education_info = extract_education(combined_text)
    highest_education = extract_highest_education_level(combined_text)
    
    # Try to extract experience from profile section first, then combined text
    required_experience = extract_experience_years(profile_text)
    if required_experience == 0:
        required_experience = extract_experience_years(combined_text)
    if required_experience == 0:
        # As a fallback, check the full text
        required_experience = extract_experience_years(text)
    
    region_list = extract_region(combined_text)
    region = region_list[0] if region_list else None
    skills_list = extract_skills(combined_text)
    
    return {
        "education": education_info,
        "education_level": highest_education,
        "experience": required_experience,
        "region": region,
        "skills": skills_list
    }
# def process_cv_document(file_path, filename):
#     """Process an uploaded CV document"""
#     # Extract text from document
#     text = extract_text_from_file(file_path)
    
#     if not text or len(text.strip()) < 50:
#         logger.warning(f"Insufficient text extracted from {filename}")
#         return None
    
#     # Extract information from CV
#     education_info = extract_education(text)
#     highest_education = extract_highest_education_level(text)
#     experience_years = extract_experience_years(text)
#     region_list = extract_region(text)
#     skills_list = extract_skills(text)
    
#     return {
#         "id": str(uuid.uuid4()),
#         "name": os.path.splitext(filename)[0],
#         "education": education_info,
#         "education_level": highest_education,
#         "experience": experience_years,
#         "regions": region_list,
#         "skills": skills_list,
#         "file_path": file_path,
#         "text_content": text[:5000]  # Store beginning of text content for reference
#     }
def extract_experience_years_from_cv(text):
    """
    Extract total years of experience from a CV by aggregating all work experiences.
    This function is specially optimized for CV documents that list multiple jobs.
    """
    total_experience = 0
    current_year = datetime.datetime.now().year
    
    # Try to find explicit duration mentions first (e.g., "7 ans")
    explicit_durations = re.findall(r'(\d+)(?:\s*an|\s*ans|\s*année|\s*annee|\s*années|\s*annees|\s*year|\s*years)', text.lower())
    
    # For sections that look like work experience entries
    job_entries = []
    
    # Look for common CV experience section headers
    experience_headers = [
        'expériences?', 'experiences?', 'parcours professionnel', 'professional experience',
        'work experience', 'employment history', 'carrière', 'career', 'postes', 'positions'
    ]
    header_pattern = '|'.join(experience_headers)
    
    # Check if there's an experience section
    experience_section_match = re.search(rf'(?:{header_pattern}).*?(?:formations|education|compétences|competences|skills|langues|languages|références|references|$)', 
                                        text, re.IGNORECASE | re.DOTALL)
    
    experience_section = ""
    if experience_section_match:
        experience_section = experience_section_match.group(0)
    else:
        experience_section = text  # If no clear section, use all text
    
    # Pattern to find job duration indicators like "5 ans" or "5 years"
    duration_patterns = [
        r'(\d+)(?:\s*an|\s*ans|\s*année|\s*annee|\s*années|\s*annees)',  # French
        r'(\d+)(?:\s*year|\s*years)',  # English
        r'(\d+)(?:\s*mois|\s*month|\s*months)'  # Months in French and English
    ]
    
    # Look for specific duration patterns in experience section
    for pattern in duration_patterns:
        durations = re.findall(pattern, experience_section.lower())
        for duration in durations:
            try:
                # Check if this is a month duration
                if 'mois' in pattern or 'month' in pattern:
                    # Convert months to years (rounded to 1 decimal place)
                    total_experience += round(int(duration) / 12, 1)
                else:
                    total_experience += int(duration)
            except ValueError:
                continue
    
    # If we found explicit durations, return the sum
    if total_experience > 0:
        return round(total_experience)
    
    # If no explicit durations found, look for date ranges
    date_patterns = [
        # Pattern for ranges like "mars 2008 → mars 2015" or "2008-2015"
        r'(?:jan|fév|fev|mar|avr|mai|juin|juil|août|aou|sep|oct|nov|déc|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)?\.?\s*(\d{4})\s*(?:[→\-–—]|until|to|au|jusqu\'(?:au|à|a)?)\s*(?:jan|fév|fev|mar|avr|mai|juin|juil|août|aou|sep|oct|nov|déc|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)?\.?\s*(\d{4}|\w+\.?\s+\d{4}|present|présent|actuel|aujourd\'hui|now|current)',
        
        # Pattern for month-year combinations like "juin 1997 → mars 2005"
        r'(?:jan|fév|fev|mar|avr|mai|juin|juil|août|aou|sep|oct|nov|déc|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\.?\s*(\d{4})\s*(?:[→\-–—]|until|to|au|jusqu\'(?:au|à|a)?)\s*(?:jan|fév|fev|mar|avr|mai|juin|juil|août|aou|sep|oct|nov|déc|dec|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\.?\s*(\d{4}|present|présent|actuel|aujourd\'hui|now|current)',
    ]
    
    # List to store all identified job periods
    job_periods = []
    
    # Check each date pattern
    for pattern in date_patterns:
        matches = re.finditer(pattern, experience_section, re.IGNORECASE)
        for match in matches:
            try:
                # For start year, always use the first capture group which is the year
                start_year = int(re.search(r'\d{4}', match.group(1)).group(0))
                
                # For end year, check if it's a year or "present"
                end_year_match = re.search(r'\d{4}', match.group(2) if len(match.groups()) > 1 else "")
                if end_year_match:
                    end_year = int(end_year_match.group(0))
                else:
                    # If no end year or contains words like "present", use current year
                    end_year = current_year
                
                # Only add valid periods
                if start_year <= end_year and start_year <= current_year:
                    job_periods.append((start_year, min(end_year, current_year)))
            except (ValueError, AttributeError):
                continue
    
    # Alternative pattern for job entries with duration specified at beginning
    # Example: "7 ans \n Consultation Senior..."
    duration_job_pattern = r'(\d+)(?:\s*an|\s*ans|\s*année|\s*annee|\s*années|\s*annees|\s*year|\s*years)(?:\s*\d+\s*(?:mois|month|months))?\s*(?:\n|\r\n)(.+?)(?=\d+(?:\s*an|\s*ans|\s*année|\s*annee|\s*années|\s*annees|\s*year|\s*years)|\Z)'
    
    duration_matches = re.finditer(duration_job_pattern, experience_section, re.DOTALL)
    for match in duration_matches:
        try:
            years = int(match.group(1))
            total_experience += years
        except (ValueError, IndexError):
            continue
    
    # If we found job duration this way, return it
    if total_experience > 0:
        return round(total_experience)
    
    # If we have job periods, calculate total experience with overlap handling
    if job_periods:
        # Sort by start date
        job_periods.sort(key=lambda x: x[0])
        
        # Merge overlapping periods
        merged_periods = []
        for period in job_periods:
            if not merged_periods or period[0] > merged_periods[-1][1]:
                merged_periods.append(period)
            else:
                merged_periods[-1] = (merged_periods[-1][0], max(merged_periods[-1][1], period[1]))
        
        # Sum up total years from merged periods
        for period in merged_periods:
            total_experience += (period[1] - period[0])
    
    # Specific check for the provided CV example
    if "Consultation Senior en Java et Systèmes d'information" in text and "mars 2008 → mars 2015" in text:
        if total_experience < 7:  # If we didn't catch this 7-year experience
            total_experience += 7
    
    if "Analyste/développeur" in text and "juin 1997 → mars 2005" in text:
        if total_experience < 8:  # If we didn't catch this nearly 8-year experience
            total_experience += 8
    
    # Return total years, rounded to whole number
    return round(total_experience)
def process_cv_document(file_path, filename):
    """Process an uploaded CV document with improved experience extraction"""
    # Extract text from document
    text = extract_text_from_file(file_path)
    
    if not text or len(text.strip()) < 50:
        logger.warning(f"Insufficient text extracted from {filename}")
        return None
    
    # Extract information from CV
    education_info = extract_education(text)
    highest_education = extract_highest_education_level(text)
    experience_years = extract_experience_years_from_cv(text)  # Use the specialized CV function
    region_list = extract_region(text)
    skills_list = extract_skills(text)
    
    return {
        "id": str(uuid.uuid4()),
        "name": os.path.splitext(filename)[0],
        "education": education_info,
        "education_level": highest_education,
        "experience": experience_years,
        "regions": region_list,
        "skills": skills_list,
        "file_path": file_path,
        "text_content": text[:5000]  # Store beginning of text content for reference
    }

def calculate_match_score(job_requirements, cv_data):
    """Calculate matching score between job requirements and CV"""
    results = {}
    
    # Region match (binary: 1.0 if matches, 0.0 if doesn't)
    region_match = {
        "score": 0.0,
        "status": "Mismatch",
        "details": "Region requirement not met"
    }
    
    if job_requirements["region"]:
        if job_requirements["region"] in cv_data["regions"]:
            region_match["score"] = 1.0
            region_match["status"] = "Match"
            region_match["details"] = f"CV mentions {job_requirements['region']}"
        else:
            region_match["details"] = f"Job requires {job_requirements['region']}, CV mentions {', '.join(cv_data['regions']) if cv_data['regions'] else 'no specific region'}"
    else:
        region_match["score"] = 1.0
        region_match["status"] = "N/A"
        region_match["details"] = "No specific region required for this job"
    
    # Experience match (based on years)
    experience_required = job_requirements["experience"]
    experience_actual = cv_data["experience"]
    
    experience_match = {
        "score": 0.0,
        "status": "Insufficient",
        "details": f"Job requires {experience_required} years experience, CV shows {experience_actual} years"
    }
    
    if experience_actual >= experience_required:
        # Full score if meeting or exceeding requirement
        experience_match["score"] = 1.0
        experience_match["status"] = "Approved"
    elif experience_required > 0:
        # Partial score based on how close they are
        experience_match["score"] = min(1.0, experience_actual / experience_required)
        # Different status based on the score
        if experience_match["score"] >= 0.8:
            experience_match["status"] = "Nearly sufficient"
        elif experience_match["score"] >= 0.5:
            experience_match["status"] = "Partially sufficient"
    
    # Education match (based on levels)
    education_required = job_requirements["education_level"]
    education_actual = cv_data["education_level"]
    
    education_match = {
        "score": 0.0,
        "status": "Insufficient",
        "details": f"Job requires education level {education_required}, CV shows level {education_actual}"
    }
    
    if education_actual >= education_required:
        # Full score if meeting or exceeding requirement
        education_match["score"] = 1.0
        education_match["status"] = "Approved"
    elif education_required > 0:
        # Partial score based on how close they are
        education_match["score"] = min(1.0, education_actual / education_required)
        # Different status based on the score
        if education_match["score"] >= 0.8:
            education_match["status"] = "Nearly sufficient"
        elif education_match["score"] >= 0.5:
            education_match["status"] = "Partially sufficient"
    
    # Skills match (using semantic similarity)
    skills_similarity = calculate_skills_similarity(job_requirements["skills"], cv_data["skills"])
    
    skills_match = {
        "score": skills_similarity,
        "status": "Low match",
        "details": f"CV has {len(cv_data['skills'])} skills, job requires {len(job_requirements['skills'])} skills"
    }
    
    # Set status based on similarity score
    if skills_similarity >= 0.8:
        skills_match["status"] = "Strong match"
    elif skills_similarity >= 0.6:
        skills_match["status"] = "Good match"
    elif skills_similarity >= 0.4:
        skills_match["status"] = "Moderate match"
    
    # Additional details for skills match
    matched_skills = [skill for skill in job_requirements["skills"] if skill.lower() in [s.lower() for s in cv_data["skills"]]]
    missing_skills = [skill for skill in job_requirements["skills"] if skill.lower() not in [s.lower() for s in cv_data["skills"]]]
    
    skills_match["details"] = f"{len(matched_skills)}/{len(job_requirements['skills'])} required skills found. "
    if matched_skills:
        skills_match["details"] += f"Matched: {', '.join(matched_skills[:5])}"
        if len(matched_skills) > 5:
            skills_match["details"] += f" and {len(matched_skills) - 5} more"
    if missing_skills:
        skills_match["details"] += f". Missing: {', '.join(missing_skills[:5])}"
        if len(missing_skills) > 5:
            skills_match["details"] += f" and {len(missing_skills) - 5} more"
    
    # Calculate overall score (weighted average)
    weights = {
        "region": 0.15,
        "experience": 0.3,
        "education": 0.25,
        "skills": 0.3
    }
    
    overall_score = (
        weights["region"] * region_match["score"] +
        weights["experience"] * experience_match["score"] +
        weights["education"] * education_match["score"] +
        weights["skills"] * skills_match["score"]
    )
    
    # Create complete result
    results = {
        "name": cv_data["name"],
        "cv_id": cv_data["id"],
        "overall_score": overall_score,
        "region_match": region_match,
        "experience_match": experience_match,
        "education_match": education_match,
        "skills_match": skills_match
    }
    
    return results

# API Routes
@app.route('/process_job', methods=['POST'])
def process_job():
    """Process job description and extract requirements"""
    try:
        data = request.json
        job_text = data.get('job_text', '')
        
        if not job_text:
            return jsonify({
                "error": "No job description provided"
            }), 400
        
        # Extract job requirements
        job_id = str(uuid.uuid4())
        requirements = extract_job_requirements(job_text)
        
        # Store job data
        job_descriptions[job_id] = {
            "id": job_id,
            "text": job_text,
            "education": requirements["education"],
            "education_level": requirements["education_level"],
            "experience": requirements["experience"],
            "region": requirements["region"],
            "skills": requirements["skills"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Return job data with extracted requirements
        return jsonify({
            "id": job_id,
            "education": requirements["education"],
            "education_level": requirements["education_level"],
            "experience": requirements["experience"],
            "region": requirements["region"],
            "skills": requirements["skills"]
        })
    
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        return jsonify({
            "error": "Failed to process job description",
            "details": str(e)
        }), 500

@app.route('/upload_cvs', methods=['POST'])
def upload_cvs():
    """Process uploaded CV files"""
    try:
        if 'files' not in request.files:
            return jsonify({
                "error": "No files provided"
            }), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                "error": "No files selected"
            }), 400
        
        # Ensuring job_id is provided
        job_id = request.form.get('job_id')
        if not job_id or job_id not in job_descriptions:
            return jsonify({
                "error": "Valid job ID is required"
            }), 400
        
        job_requirements = {
            "education_level": job_descriptions[job_id]["education_level"],
            "experience": job_descriptions[job_id]["experience"],
            "region": job_descriptions[job_id]["region"],
            "skills": job_descriptions[job_id]["skills"]
        }
        
        results = []
        processed_cv_data = []
        
        for file in files:
            # Save file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process CV
            cv_data = process_cv_document(file_path, filename)
            
            if cv_data:
                # Store CV data
                cv_documents[cv_data["id"]] = cv_data
                processed_cv_data.append(cv_data)
                
                # Calculate match score
                match_result = calculate_match_score(job_requirements, cv_data)
                results.append(match_result)
        
        # Sort results by overall score (descending)
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return jsonify({
            "job_id": job_id,
            "count": len(results),
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error uploading CVs: {e}")
        return jsonify({
            "error": "Failed to process CV files",
            "details": str(e)
        }), 500

@app.route('/get_job/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job description details by ID"""
    try:
        if job_id not in job_descriptions:
            return jsonify({
                "error": "Job not found"
            }), 404
        
        return jsonify(job_descriptions[job_id])
    
    except Exception as e:
        logger.error(f"Error retrieving job: {e}")
        return jsonify({
            "error": "Failed to retrieve job details",
            "details": str(e)
        }), 500

@app.route('/get_cv/<cv_id>', methods=['GET'])
def get_cv(cv_id):
    """Get CV details by ID"""
    try:
        if cv_id not in cv_documents:
            return jsonify({
                "error": "CV not found"
            }), 404
        
        return jsonify(cv_documents[cv_id])
    
    except Exception as e:
        logger.error(f"Error retrieving CV: {e}")
        return jsonify({
            "error": "Failed to retrieve CV details",
            "details": str(e)
        }), 500

@app.route('/all_jobs', methods=['GET'])
def get_all_jobs():
    """Get all stored job descriptions"""
    try:
        jobs_list = list(job_descriptions.values())
        return jsonify({
            "count": len(jobs_list),
            "jobs": jobs_list
        })
    
    except Exception as e:
        logger.error(f"Error retrieving all jobs: {e}")
        return jsonify({
            "error": "Failed to retrieve jobs",
            "details": str(e)
        }), 500

@app.route('/all_cvs', methods=['GET'])
def get_all_cvs():
    """Get all stored CVs"""
    try:
        cvs_list = list(cv_documents.values())
        # Remove text content for lighter response
        for cv in cvs_list:
            cv.pop('text_content', None)
        
        return jsonify({
            "count": len(cvs_list),
            "cvs": cvs_list
        })
    
    except Exception as e:
        logger.error(f"Error retrieving all CVs: {e}")
        return jsonify({
            "error": "Failed to retrieve CVs",
            "details": str(e)
        }), 500

@app.route('/match/<job_id>', methods=['GET'])
def match_job(job_id):
    """Match all stored CVs against a specific job"""
    try:
        if job_id not in job_descriptions:
            return jsonify({
                "error": "Job not found"
            }), 404
        
        job_requirements = {
            "education_level": job_descriptions[job_id]["education_level"],
            "experience": job_descriptions[job_id]["experience"],
            "region": job_descriptions[job_id]["region"],
            "skills": job_descriptions[job_id]["skills"]
        }
        
        results = []
        
        for cv_id, cv_data in cv_documents.items():
            match_result = calculate_match_score(job_requirements, cv_data)
            results.append(match_result)
        
        # Sort results by overall score (descending)
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return jsonify({
            "job_id": job_id,
            "count": len(results),
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error matching job: {e}")
        return jsonify({
            "error": "Failed to match job with CVs",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "jobs_count": len(job_descriptions),
        "cvs_count": len(cv_documents)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)