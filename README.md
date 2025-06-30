# CV-JOB Matching System

## Table of Contents
1. [Project Description](#project-description)
2. [Core Features](#core-features)
3. [Technical Implementation](#technical-implementation)
4. [Getting Started](#getting-started)
5. [System Components](#system-components)
6. [API Reference](#api-reference)
7. [Development Team](#development-team)
8. [License](#license)

---

<a id="project-description"></a>
## Project Description

CV-JOB Matching System is an AI-powered platform that automatically matches candidate CVs to job descriptions using semantic similarity and natural language processing (NLP).

This system helps recruiters:
- Analyze the job description to extract key requirements (skills, education, experience, region).
- Parse and process multiple CVs (PDF/DOCX formats).
- Score and rank candidates based on section-wise similarity to the job offer.
- Customize matching criteria with weighted preferences.

Ideal for streamlining recruitment by providing automated, explainable, and flexible CV matching.

---

<a id="core-features"></a>
## Core Features

### Document Processing
- Supports multi-format CV parsing (PDF, DOCX, plain text).
- Extracts structured job requirements from free-text job descriptions.
- Displays extracted sections: Skills, Education, Experience, and Region.

### Matching Engine
- Uses sentence-transformers for semantic similarity computation.
- Assigns a score per section and calculates a total weighted score.
- Allows recruiters to adjust importance sliders for each section.
- Accepts/rejects candidates based on a configurable threshold.

---

<a id="technical-implementation"></a>
## Technical Implementation

### Frontend
- React.js (with TypeScript)
- Next.js framework
- Tailwind CSS for styling
- Chart.js for result visualization

### Backend
- Python 3.10
- FastAPI for serving APIs
- spaCy for NLP pre-processing
- Sentence-Transformers (BERT-based models) for semantic matching

Note: A persistent database (e.g., PostgreSQL) and Redis caching system are currently not required but are part of our future plans to enable user account management, history tracking, and analytics.

---

<a id="getting-started"></a>
## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 16+

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/saadihana/CV-JOB-Matching.git
cd CV-JOB-Matching

# Backend setup
python -m venv venv
# Activate virtual environment
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate    # For Windows
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
npm run dev
```

---

<a id="system-components"></a>
## System Components

| Component         | Description                                          |
|------------------|------------------------------------------------------|
| Frontend         | React/Next.js UI for uploading CVs and job offers    |
| API Layer        | FastAPI endpoints for job parsing and CV matching    |
| Matching Engine  | Core Python service using transformer models         |
| Database         | Not implemented yet – planned for future version     |

---

<a id="api-reference"></a>
## API Reference

| Endpoint            | Method | Description                               |
|---------------------|--------|-------------------------------------------|
| /api/match          | POST   | Submit CVs and job description for analysis |
| /api/results/{id}   | GET    | Retrieve detailed match results            |
| /api/analyze        | POST   | Analyze a job description for requirements |

---

<a id="development-team"></a>
## Development Team

- SAADI Hana
- DELENDA Insaf
- BOUAZZOUNI Mohamed Amine
- LATRECHE Dhikra Maram

---

<a id="license"></a>
## License

This project is currently not under a specific license. A formal open-source license may be added in future versions.
