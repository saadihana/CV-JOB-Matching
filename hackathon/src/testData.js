// // testData.js - Contains the testing data for CV Matcher
// const testData = { 
//   "jobs": [
//     {
//       "job_id": "j001",
//       "title": "Senior Data Scientist",
//       "company": "TechInnovate Solutions",
//       "location": "Boston, MA",
//       "description": "We are seeking an experienced Data Scientist with 5+ years of expertise in machine learning and predictive modeling. The ideal candidate will have strong Python skills, experience with TensorFlow or PyTorch, and proficiency in SQL. A Master's or PhD in Computer Science, Statistics, or related field is preferred. You will be responsible for developing and implementing machine learning algorithms, analyzing large datasets, and creating data visualizations to drive business decisions. Experience with NLP and deep learning is a plus.",
//       "requirements": {
//         "education": "Master's or PhD in Computer Science, Statistics, or related field",
//         "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "SQL", "Data Visualization", "NLP"],
//         "experience": "5+ years experience in data science or machine learning",
//         "region": "Boston, MA"
//       }
//     },
//     {
//       "job_id": "j002",
//       "title": "Frontend Developer",
//       "company": "WebSolutions Inc.",
//       "location": "Seattle, WA",
//       "description": "We're looking for a talented Frontend Developer to join our team. You will be responsible for building user interfaces for web applications using React and modern JavaScript. The ideal candidate has 3+ years of experience with React, strong knowledge of CSS and responsive design, and is comfortable working with RESTful APIs. Experience with state management libraries like Redux is required. Bachelor's degree in Computer Science or equivalent experience is preferred. Our office is located in downtown Seattle, but remote work options are available for candidates in the Pacific Northwest.",
//       "requirements": {
//         "education": "Bachelor's degree in Computer Science or equivalent experience",
//         "skills": ["JavaScript", "React", "Redux", "CSS", "HTML5", "RESTful APIs", "Git"],
//         "experience": "3+ years experience in frontend development",
//         "region": "Seattle, WA or Pacific Northwest"
//       }
//     },
//     {
//       "job_id": "j003",
//       "title": "DevOps Engineer",
//       "company": "CloudTech Services",
//       "location": "Austin, TX",
//       "description": "CloudTech Services is seeking a DevOps Engineer to join our growing team. The ideal candidate has experience with AWS, Docker, Kubernetes, and CI/CD pipelines. You should have strong scripting skills in Python or Bash, and experience with infrastructure as code tools such as Terraform or CloudFormation. You will be responsible for maintaining and improving our cloud infrastructure, automating deployment processes, and ensuring system reliability. Knowledge of monitoring tools like Prometheus and Grafana is a plus. Bachelor's degree in Computer Science or equivalent experience required.",
//       "requirements": {
//         "education": "Bachelor's degree in Computer Science or equivalent experience",
//         "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Python", "Bash", "Terraform", "Linux"],
//         "experience": "4+ years experience in DevOps or Site Reliability Engineering",
//         "region": "Austin, TX"
//       }
//     }
//   ],
//   "cvs": [
//     {
//       "cv_id": "cv001",
//       "filename": "alex_johnson_resume.pdf",
//       "personal_info": {
//         "name": "Alex Johnson",
//         "email": "alex.johnson@example.com",
//         "phone": "555-123-4567",
//         "location": "Boston, MA"
//       },
//       "education": [
//         {
//           "degree": "PhD in Computer Science",
//           "institution": "Massachusetts Institute of Technology",
//           "location": "Cambridge, MA",
//           "graduation_year": 2020,
//           "details": "Specialization in Machine Learning and Artificial Intelligence"
//         },
//         {
//           "degree": "Master of Science in Statistics",
//           "institution": "University of California, Berkeley",
//           "location": "Berkeley, CA",
//           "graduation_year": 2016
//         }
//       ],
//       "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "NLP", "Deep Learning", "SQL", "R", "Data Visualization", "Git"],
//       "experience": [
//         {
//           "title": "Senior Data Scientist",
//           "company": "DataDriven Analytics",
//           "location": "Boston, MA",
//           "start_date": "2020-06",
//           "end_date": "Present",
//           "description": "Developed advanced machine learning models for customer churn prediction, improving retention by 25%. Led a team of 3 data scientists in implementing NLP solutions for customer feedback analysis. Created an automated data pipeline using Python and SQL that reduced reporting time by 60%."
//         },
//         {
//           "title": "Data Scientist",
//           "company": "TechCorp Inc.",
//           "location": "San Francisco, CA",
//           "start_date": "2017-01",
//           "end_date": "2020-05",
//           "description": "Designed and implemented machine learning algorithms for recommendation systems. Created data visualizations and dashboards for executive reporting. Collaborated with cross-functional teams to deploy models to production."
//         }
//       ],
//       "certifications": ["AWS Certified Machine Learning – Specialty", "Google Professional Data Engineer"],
//       "expected_matching": {
//         "j001": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.95,
//             "skills": 0.92,
//             "experience": 0.85,
//             "region": 1.0
//           },
//           "similarity": 0.91
//         },
//         "j002": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.80,
//             "skills": 0.20,
//             "experience": 0.30,
//             "region": 0.30
//           },
//           "similarity": 0.35,
//           "reason": "Skills mismatch"
//         },
//         "j003": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.80,
//             "skills": 0.25,
//             "experience": 0.20,
//             "region": 0.0
//           },
//           "similarity": 0.28,
//           "reason": "Skills and region mismatch"
//         }
//       }
//     },
//     {
//       "cv_id": "cv002",
//       "filename": "taylor_smith_resume.pdf",
//       "personal_info": {
//         "name": "Taylor Smith",
//         "email": "taylor.smith@example.com",
//         "phone": "555-987-6543",
//         "location": "Seattle, WA"
//       },
//       "education": [
//         {
//           "degree": "Bachelor of Science in Computer Science",
//           "institution": "University of Washington",
//           "location": "Seattle, WA",
//           "graduation_year": 2018
//         }
//       ],
//       "skills": ["JavaScript", "React", "Redux", "Node.js", "CSS", "HTML5", "REST APIs", "Responsive Design", "Git", "Jest"],
//       "experience": [
//         {
//           "title": "Senior Frontend Developer",
//           "company": "WebSolutions LLC",
//           "location": "Seattle, WA",
//           "start_date": "2021-04",
//           "end_date": "Present",
//           "description": "Lead developer for e-commerce platform rebuild using React and Redux. Improved website performance by 40% through code optimization. Implemented CI/CD workflows for frontend applications. Mentored junior developers and conducted code reviews."
//         },
//         {
//           "title": "Frontend Developer",
//           "company": "Creative Digital Agency",
//           "location": "Portland, OR",
//           "start_date": "2018-07",
//           "end_date": "2021-03",
//           "description": "Developed responsive web applications using React.js and modern JavaScript. Created UI components and implemented state management with Redux. Collaborated with designers to ensure UI/UX best practices."
//         }
//       ],
//       "certifications": ["AWS Certified Developer – Associate"],
//       "expected_matching": {
//         "j001": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.50,
//             "skills": 0.10,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.15,
//           "reason": "Skills and education mismatch"
//         },
//         "j002": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.90,
//             "skills": 0.95,
//             "experience": 0.85,
//             "region": 1.0
//           },
//           "similarity": 0.93
//         },
//         "j003": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.80,
//             "skills": 0.30,
//             "experience": 0.20,
//             "region": 0.0
//           },
//           "similarity": 0.30,
//           "reason": "Skills and region mismatch"
//         }
//       }
//     },
//     {
//       "cv_id": "cv003",
//       "filename": "jordan_rivera_resume.pdf",
//       "personal_info": {
//         "name": "Jordan Rivera",
//         "email": "jordan.rivera@example.com",
//         "phone": "555-456-7890",
//         "location": "Austin, TX"
//       },
//       "education": [
//         {
//           "degree": "Bachelor of Science in Computer Engineering",
//           "institution": "University of Texas at Austin",
//           "location": "Austin, TX",
//           "graduation_year": 2016
//         }
//       ],
//       "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "Python", "Bash", "Linux", "CI/CD", "Jenkins", "Git", "Prometheus", "Grafana"],
//       "experience": [
//         {
//           "title": "DevOps Engineer",
//           "company": "TechInfra Solutions",
//           "location": "Austin, TX",
//           "start_date": "2019-03",
//           "end_date": "Present",
//           "description": "Managed AWS infrastructure for a SaaS platform serving over 1 million users. Implemented Kubernetes clusters for container orchestration. Automated deployment processes using Jenkins and GitHub Actions. Reduced infrastructure costs by 30% through optimization."
//         },
//         {
//           "title": "Systems Administrator",
//           "company": "DataHost Inc.",
//           "location": "Dallas, TX",
//           "start_date": "2016-08",
//           "end_date": "2019-02",
//           "description": "Maintained Linux servers and implemented monitoring solutions. Assisted in cloud migration from on-premises to AWS. Scripted routine tasks to improve efficiency and reduce manual errors."
//         }
//       ],
//       "certifications": ["AWS Certified DevOps Engineer – Professional", "Certified Kubernetes Administrator"],
//       "expected_matching": {
//         "j001": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.50,
//             "skills": 0.15,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.18,
//           "reason": "Skills and education mismatch"
//         },
//         "j002": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.70,
//             "skills": 0.20,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.22,
//           "reason": "Skills and region mismatch"
//         },
//         "j003": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.85,
//             "skills": 0.95,
//             "experience": 0.90,
//             "region": 1.0
//           },
//           "similarity": 0.94
//         }
//       }
//     },
//     {
//       "cv_id": "cv004",
//       "filename": "morgan_lee_resume.pdf",
//       "personal_info": {
//         "name": "Morgan Lee",
//         "email": "morgan.lee@example.com",
//         "phone": "555-789-0123",
//         "location": "Boston, MA"
//       },
//       "education": [
//         {
//           "degree": "Master of Science in Data Science",
//           "institution": "Harvard University",
//           "location": "Cambridge, MA",
//           "graduation_year": 2019
//         },
//         {
//           "degree": "Bachelor of Science in Mathematics",
//           "institution": "Boston University",
//           "location": "Boston, MA",
//           "graduation_year": 2017
//         }
//       ],
//       "skills": ["Python", "R", "SQL", "Machine Learning", "Data Visualization", "Tableau", "Statistical Analysis", "Git"],
//       "experience": [
//         {
//           "title": "Data Scientist",
//           "company": "HealthTech Analytics",
//           "location": "Boston, MA",
//           "start_date": "2019-05",
//           "end_date": "Present",
//           "description": "Developed predictive models for patient outcomes, improving treatment efficacy by 15%. Analyzed healthcare datasets to identify cost-saving opportunities. Created interactive dashboards using Tableau for executive reporting."
//         },
//         {
//           "title": "Data Analyst",
//           "company": "Finance Solutions Corp",
//           "location": "Boston, MA",
//           "start_date": "2017-06",
//           "end_date": "2019-04",
//           "description": "Conducted financial data analysis to identify market trends. Generated weekly and monthly performance reports. Developed Excel models for financial forecasting."
//         }
//       ],
//       "certifications": ["Microsoft Certified: Data Analyst Associate"],
//       "expected_matching": {
//         "j001": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.90,
//             "skills": 0.75,
//             "experience": 0.70,
//             "region": 1.0
//           },
//           "similarity": 0.80
//         },
//         "j002": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.70,
//             "skills": 0.15,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.20,
//           "reason": "Skills mismatch"
//         },
//         "j003": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.70,
//             "skills": 0.20,
//             "experience": 0.15,
//             "region": 0.0
//           },
//           "similarity": 0.25,
//           "reason": "Skills and region mismatch"
//         }
//       }
//     },
//     {
//       "cv_id": "cv005",
//       "filename": "casey_zhang_resume.pdf",
//       "personal_info": {
//         "name": "Casey Zhang",
//         "email": "casey.zhang@example.com",
//         "phone": "555-321-6789",
//         "location": "Portland, OR"
//       },
//       "education": [
//         {
//           "degree": "Bachelor of Science in Computer Science",
//           "institution": "Portland State University",
//           "location": "Portland, OR",
//           "graduation_year": 2020
//         }
//       ],
//       "skills": ["JavaScript", "React", "CSS", "HTML5", "Node.js", "Next.js", "Tailwind CSS", "REST APIs", "Git"],
//       "experience": [
//         {
//           "title": "Frontend Developer",
//           "company": "E-Commerce Solutions",
//           "location": "Portland, OR",
//           "start_date": "2020-08",
//           "end_date": "Present",
//           "description": "Developed responsive web interfaces using React and Tailwind CSS. Implemented state management with Redux and Context API. Collaborated with UI/UX designers to create intuitive user experiences."
//         },
//         {
//           "title": "Web Development Intern",
//           "company": "Digital Agency Northwest",
//           "location": "Portland, OR",
//           "start_date": "2019-05",
//           "end_date": "2020-07",
//           "description": "Assisted in building client websites using JavaScript and CSS. Participated in code reviews and team meetings. Learned best practices for web development."
//         }
//       ],
//       "certifications": [],
//       "expected_matching": {
//         "j001": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.60,
//             "skills": 0.10,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.15,
//           "reason": "Skills and education mismatch"
//         },
//         "j002": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.90,
//             "skills": 0.85,
//             "experience": 0.60,
//             "region": 0.80
//           },
//           "similarity": 0.75
//         },
//         "j003": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.80,
//             "skills": 0.20,
//             "experience": 0.15,
//             "region": 0.0
//           },
//           "similarity": 0.25,
//           "reason": "Skills and region mismatch"
//         }
//       }
//     },
//     {
//       "cv_id": "cv006",
//       "filename": "robin_patel_resume.pdf",
//       "personal_info": {
//         "name": "Robin Patel",
//         "email": "robin.patel@example.com",
//         "phone": "555-234-5678",
//         "location": "Austin, TX"
//       },
//       "education": [
//         {
//           "degree": "Master of Science in Computer Science",
//           "institution": "University of Texas at Austin",
//           "location": "Austin, TX",
//           "graduation_year": 2017
//         },
//         {
//           "degree": "Bachelor of Engineering in Information Technology",
//           "institution": "Texas A&M University",
//           "location": "College Station, TX",
//           "graduation_year": 2015
//         }
//       ],
//       "skills": ["AWS", "GCP", "Docker", "Kubernetes", "Terraform", "Python", "Bash", "Ansible", "Jenkins", "Git", "ELK Stack"],
//       "experience": [
//         {
//           "title": "Senior DevOps Engineer",
//           "company": "CloudNative Technologies",
//           "location": "Austin, TX",
//           "start_date": "2020-01",
//           "end_date": "Present",
//           "description": "Led cloud migration projects, moving on-premises applications to AWS. Designed and implemented CI/CD pipelines using Jenkins and GitHub Actions. Reduced deployment time from days to hours through automation."
//         },
//         {
//           "title": "Cloud Engineer",
//           "company": "SoftInfra Inc.",
//           "location": "Houston, TX",
//           "start_date": "2017-08",
//           "end_date": "2019-12",
//           "description": "Managed cloud infrastructure on AWS and GCP. Implemented infrastructure as code using Terraform. Automated server provisioning and configuration using Ansible."
//         }
//       ],
//       "certifications": ["AWS Certified Solutions Architect – Professional", "Google Cloud Professional Cloud Architect"],
//       "expected_matching": {
//         "j001": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.75,
//             "skills": 0.20,
//             "experience": 0.15,
//             "region": 0.0
//           },
//           "similarity": 0.25,
//           "reason": "Skills mismatch"
//         },
//         "j002": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.75,
//             "skills": 0.20,
//             "experience": 0.15,
//             "region": 0.0
//           },
//           "similarity": 0.25,
//           "reason": "Skills and region mismatch"
//         },
//         "j003": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.90,
//             "skills": 0.90,
//             "experience": 0.95,
//             "region": 1.0
//           },
//           "similarity": 0.92
//         }
//       }
//     },
//     {
//       "cv_id": "cv007",
//       "filename": "sam_martinez_resume.pdf",
//       "personal_info": {
//         "name": "Sam Martinez",
//         "email": "sam.martinez@example.com",
//         "phone": "555-876-5432",
//         "location": "San Francisco, CA"
//       },
//       "education": [
//         {
//           "degree": "Master of Science in Machine Learning",
//           "institution": "Stanford University",
//           "location": "Stanford, CA",
//           "graduation_year": 2021
//         },
//         {
//           "degree": "Bachelor of Science in Computer Science",
//           "institution": "University of California, Los Angeles",
//           "location": "Los Angeles, CA",
//           "graduation_year": 2019
//         }
//       ],
//       "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "Computer Vision", "SQL", "Docker", "Git"],
//       "experience": [
//         {
//           "title": "Machine Learning Engineer",
//           "company": "AI Solutions Inc.",
//           "location": "San Francisco, CA",
//           "start_date": "2021-06",
//           "end_date": "Present",
//           "description": "Developed computer vision models for object detection and tracking. Optimized neural network performance for edge devices. Collaborated with research team to implement state-of-the-art algorithms."
//         },
//         {
//           "title": "ML Research Intern",
//           "company": "Research Labs",
//           "location": "Palo Alto, CA",
//           "start_date": "2020-01",
//           "end_date": "2021-05",
//           "description": "Conducted research on deep learning approaches for natural language processing. Published paper on efficient transformer architectures. Implemented prototype systems for text classification."
//         }
//       ],
//       "certifications": ["NVIDIA Deep Learning Institute - Deep Learning Fundamentals"],
//       "expected_matching": {
//         "j001": {
//           "status": "Accepted",
//           "section_scores": {
//             "education": 0.85,
//             "skills": 0.85,
//             "experience": 0.75,
//             "region": 0.0
//           },
//           "similarity": 0.70
//         },
//         "j002": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.75,
//             "skills": 0.20,
//             "experience": 0.10,
//             "region": 0.0
//           },
//           "similarity": 0.20,
//           "reason": "Skills and region mismatch"
//         },
//         "j003": {
//           "status": "Rejected",
//           "section_scores": {
//             "education": 0.75,
//             "skills": 0.30,
//             "experience": 0.15,
//             "region": 0.0
//           },
//           "similarity": 0.28,
//           "reason": "Skills and region mismatch"
//         }
//       }
//     }
//   ]
// };

// export default testData;