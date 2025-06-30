# CV-JOB Matching System

## Table of Contents
1. [Project Description](#project-description)
2. [Core Features](#core-features)
3. [Technical Implementation](#technical-implementation)
4. [Getting Started](#getting-started)
5. [System Components](#system-components)
6. [API Reference](#api-reference)
7. [Development Team](#development-team)
8. [License Information](#license-information)

---

<a id="project-description"></a>
## Project Description
An AI-powered platform that automatically matches candidate CVs with relevant job opportunities using natural language processing.

The system:
- Parses resume content (PDF/DOCX formats)
- Analyzes job description requirements
- Calculates compatibility scores
- Provides ranked matching results

---

<a id="core-features"></a>
## Core Features

### Document Processing
- Multi-format CV parsing (PDF, DOCX, plain text)
- Job description requirement extraction

### Matching Engine
- Semantic matching using transformer models
- Explainable scoring system
- Customizable matching criteria

---

<a id="technical-implementation"></a>
## Technical Implementation

### Frontend
- React.js with TypeScript
- Next.js framework
- Tailwind CSS
- Chart.js visualizations

### Backend
- Python 3.10
- FastAPI framework
- spaCy NLP processing
- Sentence-Transformers

---

<a id="getting-started"></a>
## Getting Started

### Requirements
- Python 3.10+
- Node.js 16+
- PostgreSQL 14+

### Installation
```bash
git clone https://github.com/saadihana/CV-JOB-Matching.git
cd CV-JOB-Matching
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
cd frontend
npm install
npm run dev

<a id="system-components"></a>

System Components
Frontend: Next.js application

API Layer: FastAPI endpoints

Matching Service: Python engine

Database: PostgreSQL with Redis cache
```

<a id="api-reference"></a>

API Reference
Endpoint	Method	Description
/api/match	POST	Submit CV+job for analysis
/api/results/{id}	GET	Retrieve matching results
/api/analyze	POST	Job description parsing
<a id="development-team"></a>

Development Team
SAADI Hana 

DELENDA Insaf 

BOUAZZOUNI Mohamed Amine 

LATRECHE Dhikra Maram
