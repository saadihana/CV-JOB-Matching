// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [jobDesc, setJobDesc] = useState("");
//   const [files, setFiles] = useState([]);
//   const [results, setResults] = useState([]);
//   const [educationWeight, setEducationWeight] = useState(1);
//   const [skillsWeight, setSkillsWeight] = useState(1);
//   const [experienceWeight, setExperienceWeight] = useState(1);
//   const [regionWeight, setRegionWeight] = useState(1);
//   const [loading, setLoading] = useState(false);
//   const [expandedResult, setExpandedResult] = useState(null);

//   const handleUpload = async () => {
//     if (!jobDesc || files.length === 0) return;

//     const formData = new FormData();
//     formData.append("job_description", jobDesc);
//     formData.append("education_weight", educationWeight);
//     formData.append("skills_weight", skillsWeight);
//     formData.append("experience_weight", experienceWeight);
//     formData.append("region_weight", regionWeight);
    
//     for (let file of files) {
//       formData.append("files", file);
//     }

//     setLoading(true);
//     try {
//       const res = await axios.post("http://localhost:8000/match-cvs/", formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });
//       setResults(res.data.ranked_cvs || []);
//     } catch (err) {
//       console.error("Upload failed:", err);
//       alert("Error: " + (err.response?.data?.error || "Unknown error occurred"));
//     } finally {
//       setLoading(false);
//     }
//   };

//   const toggleExpand = (index) => {
//     if (expandedResult === index) {
//       setExpandedResult(null);
//     } else {
//       setExpandedResult(index);
//     }
//   };

//   const acceptedCVs = results.filter((res) => res.status === "Accepted");
//   const rejectedCVs = results.filter((res) => res.status === "Rejected");

//   // Get color based on weight value
//   const getWeightColor = (value) => {
//     if (value < 0.7) return "#6c757d"; // gray for low
//     if (value < 1.5) return "#0d6efd"; // green for medium
//     if (value < 2.3) return "#fd7e14"; // orange for high
//     return "#dc3545"; // red for very high
//   };

//   return (
//     <div style={styles.container}>
      
//       <h1 style={styles.title}></h1>

//       <div style={styles.section}>
//         <h3 style={styles.sectionTitle}>Job Information</h3>
//         <label style={styles.label}>Job Description:</label>
//         <textarea
//           rows="6"
//           placeholder="Paste the job description here..."
//           value={jobDesc}
//           onChange={(e) => setJobDesc(e.target.value)}
//           style={styles.textarea}
//         />
//       </div>

//       <div style={styles.section}>
//         <h3 style={styles.sectionTitle}>Sections Importance</h3>
//         <p style={styles.infoText}>
//           Slide to adjust importance of each factor in CV matching
//         </p>
        
//         <div style={styles.weightsGrid}>
//           <div style={styles.sliderItem}>
//             <div style={styles.sliderHeader}>
//               <label style={styles.sliderLabel}>Education</label>
//               <span style={{...styles.sliderValue, backgroundColor: getWeightColor(educationWeight)}}>{educationWeight.toFixed(1)}</span>
//             </div>
//             <div style={styles.sliderContainer}>
//               <input
//                 type="range"
//                 min="0"
//                 max="3"
//                 step="0.1"
//                 value={educationWeight}
//                 onChange={(e) => setEducationWeight(parseFloat(e.target.value))}
//                 style={{
//                   ...styles.slider,
//                   background: `linear-gradient(to right, ${getWeightColor(educationWeight)} 0%, ${getWeightColor(educationWeight)} ${(educationWeight/3)*100}%, #ddd ${(educationWeight/3)*100}%, #ddd 100%)`
//                 }}
//               />
//             </div>
//           </div>
          
//           <div style={styles.sliderItem}>
//             <div style={styles.sliderHeader}>
//               <label style={styles.sliderLabel}>Skills</label>
//               <span style={{...styles.sliderValue, backgroundColor: getWeightColor(skillsWeight)}}>{skillsWeight.toFixed(1)}</span>
//             </div>
//             <div style={styles.sliderContainer}>
//               <input
//                 type="range"
//                 min="0"
//                 max="3"
//                 step="0.1"
//                 value={skillsWeight}
//                 onChange={(e) => setSkillsWeight(parseFloat(e.target.value))}
//                 style={{
//                   ...styles.slider,
//                   background: `linear-gradient(to right, ${getWeightColor(skillsWeight)} 0%, ${getWeightColor(skillsWeight)} ${(skillsWeight/3)*100}%, #ddd ${(skillsWeight/3)*100}%, #ddd 100%)`
//                 }}
//               />
//             </div>
//           </div>
          
//           <div style={styles.sliderItem}>
//             <div style={styles.sliderHeader}>
//               <label style={styles.sliderLabel}>Experience</label>
//               <span style={{...styles.sliderValue, backgroundColor: getWeightColor(experienceWeight)}}>{experienceWeight.toFixed(1)}</span>
//             </div>
//             <div style={styles.sliderContainer}>
//               <input
//                 type="range"
//                 min="0"
//                 max="3"
//                 step="0.1"
//                 value={experienceWeight}
//                 onChange={(e) => setExperienceWeight(parseFloat(e.target.value))}
//                 style={{
//                   ...styles.slider,
//                   background: `linear-gradient(to right, ${getWeightColor(experienceWeight)} 0%, ${getWeightColor(experienceWeight)} ${(experienceWeight/3)*100}%, #ddd ${(experienceWeight/3)*100}%, #ddd 100%)`
//                 }}
//               />
//             </div>
//           </div>
          
//           <div style={styles.sliderItem}>
//             <div style={styles.sliderHeader}>
//               <label style={styles.sliderLabel}>Region</label>
//               <span style={{...styles.sliderValue, backgroundColor: getWeightColor(regionWeight)}}>{regionWeight.toFixed(1)}</span>
//             </div>
//             <div style={styles.sliderContainer}>
//               <input
//                 type="range"
//                 min="0"
//                 max="3"
//                 step="0.1"
//                 value={regionWeight}
//                 onChange={(e) => setRegionWeight(parseFloat(e.target.value))}
//                 style={{
//                   ...styles.slider,
//                   background: `linear-gradient(to right, ${getWeightColor(regionWeight)} 0%, ${getWeightColor(regionWeight)} ${(regionWeight/3)*100}%, #ddd ${(regionWeight/3)*100}%, #ddd 100%)`
//                 }}
//               />
//             </div>
//           </div>
//         </div>

//         <div style={styles.weightLegend}>
//           <div style={styles.legendItem}>
//             <span style={{...styles.legendColor, backgroundColor: "#6c757d"}}></span>
//             <span>Low</span>
//           </div>
//           <div style={styles.legendItem}>
//             <span style={{...styles.legendColor, backgroundColor: "#0d6efd"}}></span>
//             <span>Medium</span>
//           </div>
//           <div style={styles.legendItem}>
//             <span style={{...styles.legendColor, backgroundColor: "#fd7e14"}}></span>
//             <span>High</span>
//           </div>
//           <div style={styles.legendItem}>
//             <span style={{...styles.legendColor, backgroundColor: "#dc3545"}}></span>
//             <span>Very High</span>
//           </div>
//         </div>
//       </div>

//       <div style={styles.section}>
//         <h3 style={styles.sectionTitle}>CV Upload</h3>
//         <label style={styles.label}>Upload CVs (PDF, DOCX, TXT):</label>
//         <input
//           type="file"
//           multiple
//           accept=".pdf,.docx,.txt"
//           onChange={(e) => setFiles(e.target.files)}
//           style={styles.input}
//         />
//       </div>

//       <button
//         onClick={handleUpload}
//         disabled={!jobDesc || files.length === 0 || loading}
//         style={{
//           ...styles.button,
//           backgroundColor: loading ? "#999" : "#22c55e",
//           cursor: loading ? "not-allowed" : "pointer",
//         }}
//       >
//         {loading ? "Matching..." : "Match CVs"}
//       </button>

//       {results.length > 0 && (
//         <div style={styles.resultsContainer}>
//           {acceptedCVs.length > 0 && (
//             <div style={styles.section}>
//               <h3 style={styles.resultsTitle}>Accepted CVs</h3>
//               <ul style={styles.resultsList}>
//                 {acceptedCVs.map((res, idx) => (
//                   <li key={idx} style={styles.resultItem}>
//                     <div style={styles.resultHeader} onClick={() => toggleExpand(idx)}>
//                       <div style={styles.resultBasicInfo}>
//                         <span style={styles.rank}>{idx + 1}.</span>{" "}
//                         <strong>{res.filename}</strong>
//                       </div>
//                       <div style={styles.resultScores}>
//                         <span style={styles.resultScore}>
//                           <strong>Overall:</strong> {res.similarity.toFixed(2)}
//                         </span>
//                         <a
//                           href={`http://localhost:8000${res.file_path}`}
//                           download={res.filename}
//                           style={styles.downloadLink}
//                           onClick={(e) => e.stopPropagation()}
//                         >
//                           {/* Download CV */}
//                         </a>
//                       </div>
//                     </div>
                    
//                     {expandedResult === idx && res.section_scores && (
//                       <div style={styles.detailedScores}>
//                         <h4 style={styles.detailedScoresTitle}>Section Scores:</h4>
//                         <div style={styles.scoreGrid}>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Education:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.education.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Skills:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.skills.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Experience:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.experience.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Region:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.region.toFixed(2)}
//                             </span>
//                           </div>
//                         </div>
//                       </div>
//                     )}
//                   </li>
//                 ))}
//               </ul>
//             </div>
//           )}

//           {rejectedCVs.length > 0 && (
//             <div style={styles.section}>
//               <h3 style={styles.resultsTitle}>Rejected CVs</h3>
//               <ul style={styles.resultsList}>
//                 {rejectedCVs.map((res, idx) => (
//                   <li key={idx} style={styles.resultItem}>
//                     <div style={styles.resultHeader} onClick={() => toggleExpand(acceptedCVs.length + idx)}>
//                       <div style={styles.resultBasicInfo}>
//                         <span style={styles.rank}>{idx + 1}.</span>{" "}
//                         <strong>{res.filename}</strong>
//                       </div>
//                       <div style={styles.resultScores}>
//                         <span style={styles.resultScore}>
//                           <strong>Overall:</strong> {res.similarity.toFixed(2)}
//                         </span>
//                         <span style={styles.rejectionReason}>
//                           {res.reason}
//                         </span>
//                       </div>
//                     </div>
                    
//                     {expandedResult === acceptedCVs.length + idx && res.section_scores && (
//                       <div style={styles.detailedScores}>
//                         <h4 style={styles.detailedScoresTitle}>Section Scores:</h4>
//                         <div style={styles.scoreGrid}>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Education:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.education.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Skills:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.skills.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Experience:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.experience.toFixed(2)}
//                             </span>
//                           </div>
//                           <div style={styles.scoreItem}>
//                             <span style={styles.scoreLabel}>Region:</span>
//                             <span style={styles.scoreValue}>
//                               {res.section_scores.region.toFixed(2)}
//                             </span>
//                           </div>
//                         </div>
//                       </div>
//                     )}
//                   </li>
//                 ))}
//               </ul>
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }


// const styles = {
//   container: {
//     padding: "2rem",
//     maxWidth: "900px",
//     margin: "auto",
//     backgroundColor: "#fff",
//     boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
//     borderRadius: "8px",
//   },
//   title: {
//     textAlign: "center",
//     fontSize: "2.5rem",
//     color: '#22c55e',
//     marginBottom: "1.5rem",
//     fontWeight: "600",
//   },
//   section: {
//     marginBottom: "2rem",
//     padding: "1.5rem",
//     backgroundColor: "#f8f9fa",
//     borderRadius: "8px",
//   },
//   sectionTitle: {
//     fontSize: "1.25rem",
//     color: '#22c55e',
//     marginBottom: "1rem",
//     fontWeight: "600",
//   },
//   label: {
//     display: "block",
//     marginTop: "1rem",
//     fontSize: "1rem",
//     fontWeight: "600",
//     color: "#333",
//   },
//   textarea: {
//     width: "100%",
//     padding: "1rem",
//     borderRadius: "8px",
//     border: "1px solid #ddd",
//     marginTop: "0.5rem",
//     resize: "vertical",
//     fontSize: "1rem",
//     transition: "border-color 0.3s ease",
//   },
//   input: {
//     display: "block",
//     width: "100%",
//     marginTop: "0.5rem",
//     padding: "0.75rem",
//     fontSize: "1rem",
//     borderRadius: "8px",
//     border: "1px solid #ddd",
//     backgroundColor: "#fff",
//     transition: "background-color 0.3s ease",
//   },
//   infoText: {
//     fontSize: "0.9rem",
//     color: "#666",
//     marginBottom: "1rem",
//   },
//   // New compact grid layout for sliders
//   weightsGrid: {
//     display: "grid",
//     gridTemplateColumns: "repeat(2, 1fr)",
//     gap: "1rem",
//     marginBottom: "1rem",
//   },
//   sliderItem: {
//     padding: "0.75rem",
//     backgroundColor: "#fff",
//     borderRadius: "6px",
//     boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
//   },
//   sliderHeader: {
//     display: "flex",
//     justifyContent: "space-between",
//     alignItems: "center",
//     marginBottom: "0.5rem",
//   },
//   sliderLabel: {
//     fontSize: "0.9rem",
//     fontWeight: "600",
//     color: "#333",
//   },
//   sliderValue: {
//     fontSize: "0.8rem",
//     fontWeight: "600",
//     color: "#fff",
//     padding: "0.15rem 0.4rem",
//     borderRadius: "4px",
//     minWidth: "2rem",
//     textAlign: "center",
//   },
//   sliderContainer: {
//     width: "100%",
//     position: "relative",
//   },
//   slider: {
//     width: "100%",
//     height: "6px",
//     borderRadius: "3px",
//     appearance: "none",
//     outline: "none",
//     cursor: "pointer",
//     transition: "background 0.2s",
//     "&::-webkit-slider-thumb": {
//       appearance: "none",
//       width: "16px",
//       height: "16px",
//       borderRadius: "50%",
//       background: "#fff",
//       border: "2px solid #22c55e",
//       cursor: "pointer",
//     },
//     "&::-moz-range-thumb": {
//       width: "16px",
//       height: "16px",
//       borderRadius: "50%",
//       background: "#fff",
//       border: "2px solid #22c55e",
//       cursor: "pointer",
//     },
//   },
//   // Color legend
//   weightLegend: {
//     display: "flex",
//     justifyContent: "center",
//     gap: "1rem",
//     marginTop: "0.5rem",
//   },
//   legendItem: {
//     display: "flex",
//     alignItems: "center",
//     gap: "0.3rem",
//     fontSize: "0.75rem",
//     color: "#666",
//   },
//   legendColor: {
//     width: "10px",
//     height: "10px",
//     borderRadius: "50%",
//     display: "inline-block",
//   },
//   button: {
//     padding: "0.75rem 1.5rem",
//     backgroundColor: '#22c55e',
//     color: "#fff",
//     border: "none",
//     borderRadius: "8px",
//     fontWeight: "600",
//     cursor: "pointer",
//     transition: "background-color 0.3s ease",
//     fontSize: "1rem",
//     width: "100%",
//     marginTop: "1rem",
//   },
//   resultsContainer: {
//     marginTop: "2rem",
//     paddingTop: "1rem",
//     borderTop: "1px solid #ddd",
//   },
//   resultsTitle: {
//     fontSize: "1.25rem",
//     color: '#22c55e',
//     marginBottom: "1rem",
//     fontWeight: "600",
//   },
//   resultsList: {
//     listStyleType: "none",
//     paddingLeft: "0",
//   },
//   resultItem: {
//     marginBottom: "0.75rem",
//     borderRadius: "8px",
//     overflow: "hidden",
//     boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
//     backgroundColor: "#fff",
//   },
//   resultHeader: {
//     padding: "1rem",
//     display: "flex",
//     justifyContent: "space-between",
//     alignItems: "center",
//     cursor: "pointer",
//     transition: "background-color 0.2s ease",
//     backgroundColor: "#f8f9fa",
//     "&:hover": {
//       backgroundColor: "#e9ecef",
//     },
//   },
//   resultBasicInfo: {
//     display: "flex",
//     alignItems: "center",
//   },
//   resultScores: {
//     display: "flex",
//     alignItems: "center",
//     gap: "1rem",
//   },
//   resultScore: {
//     fontSize: "0.9rem",
//     color: "#333",
//   },
//   rank: {
//     fontWeight: "600",
//     color: '#22c55e',
//     marginRight: "0.5rem",
//   },
//   downloadLink: {
//     color: '#22c55e',
//     textDecoration: "none",
//     fontWeight: "600",
//     padding: "0.25rem 0.5rem",
//     backgroundColor: "#e6f2ff",
//     borderRadius: "4px",
//     fontSize: "0.8rem",
//   },
//   rejectionReason: {
//     color: "#dc3545",
//     fontStyle: "italic",
//     fontSize: "0.8rem",
//     padding: "0.25rem 0.5rem",
//     backgroundColor: "#ffe6e6",
//     borderRadius: "4px",
//   },
//   detailedScores: {
//     padding: "1rem",
//     backgroundColor: "#fff",
//     borderTop: "1px solid #eee",
//   },
//   detailedScoresTitle: {
//     fontSize: "1rem",
//     color: "#333",
//     marginBottom: "0.75rem",
//     fontWeight: "600",
//   },
//   scoreGrid: {
//     display: "grid",
//     gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
//     gap: "1rem",
//   },
//   scoreItem: {
//     padding: "0.75rem",
//     backgroundColor: "#f8f9fa",
//     borderRadius: "6px",
//     display: "flex",
//     flexDirection: "column",
//   },
//   scoreLabel: {
//     fontSize: "0.8rem",
//     color: "#666",
//     marginBottom: "0.25rem",
//   },
//   scoreValue: {
//     fontSize: "1.1rem",
//     fontWeight: "600",
//     color: '#22c55e',
//   },
  
// };

// export default App;



// import React, { useState, useRef } from 'react';
// import axios from 'axios';
// import { 
//   Container, Box, Typography, Grid, Paper, TextField, Button, 
//   CircularProgress, Table, TableBody, TableCell, TableContainer, 
//   TableHead, TableRow, Chip, Collapse, IconButton, List, ListItem, 
//   ListItemText, Divider, Alert, Card, CardContent
// } from '@mui/material';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// import CloudUploadIcon from '@mui/icons-material/CloudUpload';
// import WorkIcon from '@mui/icons-material/Work';
// import PersonIcon from '@mui/icons-material/Person';
// import CheckCircleIcon from '@mui/icons-material/CheckCircle';
// import ErrorIcon from '@mui/icons-material/Error';
// import InfoIcon from '@mui/icons-material/Info';

// // API URL
// const API_URL = 'http://localhost:5000';

// function App() {
//   // State for job description
//   const [jobText, setJobText] = useState('');
//   const [jobData, setJobData] = useState(null);
//   const [jobLoading, setJobLoading] = useState(false);
//   const [jobError, setJobError] = useState(null);

//   // State for CV upload
//   const fileInput = useRef(null);
//   const [selectedFiles, setSelectedFiles] = useState([]);
//   const [uploadedCVs, setUploadedCVs] = useState([]);
//   const [cvLoading, setCVLoading] = useState(false);
//   const [cvError, setCVError] = useState(null);

//   // State for results
//   const [matchResults, setMatchResults] = useState([]);
//   const [matchLoading, setMatchLoading] = useState(false);
//   const [matchError, setMatchError] = useState(null);
//   const [expandedRows, setExpandedRows] = useState({});

//   // Process job description
//   const handleJobSubmit = async (e) => {
//     e.preventDefault();
//     if (!jobText.trim()) return;

//     setJobLoading(true);
//     setJobError(null);

//     try {
//       const response = await axios.post(`${API_URL}/process_job`, {
//         job_text: jobText
//       });
//       setJobData(response.data);
//     } catch (error) {
//       console.error('Error processing job description:', error);
//       setJobError('Failed to process job description. Please try again.');
//     } finally {
//       setJobLoading(false);
//     }
//   };

//   // Handle file selection
//   const handleFileChange = (e) => {
//     const files = Array.from(e.target.files);
//     setSelectedFiles(files);
//   };

//   // Upload and process CVs
//   const handleCVUpload = async () => {
//     if (selectedFiles.length === 0) return;

//     setCVLoading(true);
//     setCVError(null);

//     const formData = new FormData();
//     selectedFiles.forEach(file => {
//       formData.append('cvs', file);
//     });

//     try {
//       const response = await axios.post(`${API_URL}/upload_cvs`, formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });
//       setUploadedCVs(response.data.cvs);
//       setSelectedFiles([]);
//     } catch (error) {
//       console.error('Error uploading CVs:', error);
//       setCVError('Failed to upload and process CVs. Please check file formats and try again.');
//     } finally {
//       setCVLoading(false);
//     }
//   };

//   // Match CVs with job
//   const handleMatchProcess = async () => {
//     if (!jobData || uploadedCVs.length === 0) return;

//     setMatchLoading(true);
//     setMatchError(null);

//     try {
//       const response = await axios.post(`${API_URL}/match`, {
//         job_id: jobData.id,
//         cv_ids: uploadedCVs.map(cv => cv.id)
//       });
//       setMatchResults(response.data.results);
//     } catch (error) {
//       console.error('Error matching CVs with job:', error);
//       setMatchError('Failed to match CVs with job description. Please try again.');
//     } finally {
//       setMatchLoading(false);
//     }
//   };

//   // Toggle expanded row details
//   const toggleRow = (id) => {
//     setExpandedRows(prev => ({
//       ...prev,
//       [id]: !prev[id]
//     }));
//   };

//   // Get color based on score
//   const getScoreColor = (score) => {
//     if (score >= 0.7) return 'success';
//     if (score >= 0.5) return 'warning';
//     return 'error';
//   };

//   return (
//     <Container maxWidth="lg" sx={{ py: 4 }}>
//       <Typography variant="h3" align="center" gutterBottom sx={{ mb: 4 }}>
//         CV-Job Matcher
//       </Typography>

//       <Grid container spacing={3}>
//         {/* Job Description Section */}
//         <Grid item xs={12} md={6}>
//           <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
//             <Box display="flex" alignItems="center" mb={2}>
//               <WorkIcon color="primary" sx={{ mr: 1 }} />
//               <Typography variant="h5">Job Description</Typography>
//             </Box>
//             <Divider sx={{ mb: 2 }} />
            
//             <form onSubmit={handleJobSubmit}>
//               <TextField
//                 label="Paste job description here"
//                 multiline
//                 rows={10}
//                 fullWidth
//                 value={jobText}
//                 onChange={(e) => setJobText(e.target.value)}
//                 margin="normal"
//                 variant="outlined"
//                 placeholder="Include sections like 'Mission' and 'Profil' as in Mobilis job descriptions..."
//               />
              
//               <Button 
//                 type="submit" 
//                 variant="contained" 
//                 color="primary"
//                 disabled={jobLoading || !jobText.trim()}
//                 sx={{ mt: 2 }}
//                 startIcon={jobLoading ? <CircularProgress size={20} /> : null}
//               >
//                 {jobLoading ? 'Processing...' : 'Extract Requirements'}
//               </Button>
              
//               {jobError && (
//                 <Alert severity="error" sx={{ mt: 2 }}>{jobError}</Alert>
//               )}
//             </form>
            
//             {jobData && (
//               <Box mt={3}>
//                 <Typography variant="h6" gutterBottom>Extracted Requirements:</Typography>
//                 <Card variant="outlined" sx={{ mb: 2 }}>
//                   <CardContent>
//                     <Typography variant="subtitle1" color="primary">Education</Typography>
//                     <Typography variant="body2" sx={{ mb: 2 }}>{jobData.education || 'Not specified'}</Typography>
                    
//                     <Typography variant="subtitle1" color="primary">Experience</Typography>
//                     <Typography variant="body2" sx={{ mb: 2 }}>{jobData.experience} years</Typography>
                    
//                     <Typography variant="subtitle1" color="primary">Region</Typography>
//                     <Typography variant="body2" sx={{ mb: 2 }}>{jobData.region || 'Not specified'}</Typography>
                    
//                     <Typography variant="subtitle1" color="primary">Skills</Typography>
//                     <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
//                       {jobData.skills.length > 0 ? (
//                         jobData.skills.map((skill, idx) => (
//                           <Chip key={idx} label={skill} size="small" />
//                         ))
//                       ) : (
//                         <Typography variant="body2">No specific skills extracted</Typography>
//                       )}
//                     </Box>
//                   </CardContent>
//                 </Card>
//               </Box>
//             )}
//           </Paper>
//         </Grid>

//         {/* CV Upload Section */}
//         <Grid item xs={12} md={6}>
//           <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
//             <Box display="flex" alignItems="center" mb={2}>
//               <PersonIcon color="primary" sx={{ mr: 1 }} />
//               <Typography variant="h5">Upload CVs</Typography>
//             </Box>
//             <Divider sx={{ mb: 2 }} />
            
//             <Box textAlign="center" mb={3}>
//               <input
//                 type="file"
//                 multiple
//                 accept=".pdf,.doc,.docx,.txt"
//                 style={{ display: 'none' }}
//                 ref={fileInput}
//                 onChange={handleFileChange}
//               />
//               <Button
//                 variant="outlined"
//                 startIcon={<CloudUploadIcon />}
//                 onClick={() => fileInput.current.click()}
//                 sx={{ mb: 2 }}
//                 fullWidth
//               >
//                 Select CV Files (PDF, DOC, DOCX, TXT)
//               </Button>
              
//               {selectedFiles.length > 0 && (
//                 <Box mb={2}>
//                   <Typography variant="body2" sx={{ mb: 1 }}>
//                     {selectedFiles.length} file(s) selected
//                   </Typography>
//                   <List dense sx={{ maxHeight: '200px', overflow: 'auto', bgcolor: '#f5f5f5', borderRadius: 1 }}>
//                     {selectedFiles.map((file, index) => (
//                       <ListItem key={index} divider={index < selectedFiles.length - 1}>
//                         <ListItemText 
//                           primary={file.name} 
//                           secondary={`${(file.size / 1024).toFixed(1)} KB`} 
//                         />
//                       </ListItem>
//                     ))}
//                   </List>
                  
//                   <Button
//                     variant="contained"
//                     color="primary"
//                     onClick={handleCVUpload}
//                     disabled={cvLoading}
//                     sx={{ mt: 2 }}
//                     fullWidth
//                     startIcon={cvLoading ? <CircularProgress size={20} /> : null}
//                   >
//                     {cvLoading ? 'Processing...' : 'Upload and Process CVs'}
//                   </Button>
//                 </Box>
//               )}
              
//               {cvError && (
//                 <Alert severity="error" sx={{ mt: 2 }}>{cvError}</Alert>
//               )}
//             </Box>
            
//             {uploadedCVs.length > 0 && (
//               <Box mt={2}>
//                 <Typography variant="h6" gutterBottom>Processed CVs ({uploadedCVs.length})</Typography>
//                 <List dense sx={{ maxHeight: '200px', overflow: 'auto', bgcolor: '#f5f5f5', borderRadius: 1 }}>
//                   {uploadedCVs.map((cv, index) => (
//                     <ListItem key={index} divider={index < uploadedCVs.length - 1}>
//                       <ListItemText 
//                         primary={cv.name} 
//                         secondary={`Experience: ${cv.experience} years | Education: ${cv.education.substring(0, 30)}${cv.education.length > 30 ? '...' : ''}`} 
//                       />
//                     </ListItem>
//                   ))}
//                 </List>
                
//                 <Button
//                   variant="contained"
//                   color="secondary"
//                   onClick={handleMatchProcess}
//                   disabled={!jobData || matchLoading}
//                   sx={{ mt: 2 }}
//                   fullWidth
//                   startIcon={matchLoading ? <CircularProgress size={20} /> : null}
//                 >
//                   {matchLoading ? 'Processing...' : 'Match CVs with Job'}
//                 </Button>
                
//                 {matchError && (
//                   <Alert severity="error" sx={{ mt: 2 }}>{matchError}</Alert>
//                 )}
//               </Box>
//             )}
//           </Paper>
//         </Grid>

//         {/* Results Section */}
//         {matchResults.length > 0 && (
//           <Grid item xs={12}>
//             <Paper elevation={3} sx={{ p: 3 }}>
//               <Box display="flex" alignItems="center" mb={2}>
//                 <CheckCircleIcon color="primary" sx={{ mr: 1 }} />
//                 <Typography variant="h5">Matching Results</Typography>
//               </Box>
//               <Divider sx={{ mb: 2 }} />
              
//               <Typography variant="body2" gutterBottom>
//                 {matchResults.length} candidates ranked by matching score
//               </Typography>
              
//               <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}>
//                 <Table>
//                   <TableHead>
//                     <TableRow sx={{ bgcolor: '#f5f5f5' }}>
//                       <TableCell>Candidate</TableCell>
//                       <TableCell>Overall Match</TableCell>
//                       <TableCell>Region</TableCell>
//                       <TableCell>Experience</TableCell>
//                       <TableCell>Education</TableCell>
//                       <TableCell>Skills</TableCell>
//                       <TableCell>Details</TableCell>
//                     </TableRow>
//                   </TableHead>
//                   <TableBody>
//                     {matchResults.map((result, index) => (
//                       <React.Fragment key={index}>
//                         <TableRow>
//                           <TableCell><strong>{result.name}</strong></TableCell>
//                           <TableCell>
//                             <Chip 
//                               label={`${(result.overall_score * 100).toFixed(0)}%`}
//                               color={getScoreColor(result.overall_score)}
//                               variant="filled"
//                             />
//                           </TableCell>
//                           <TableCell>
//                             <Chip 
//                               label={result.region_match.status}
//                               color={result.region_match.score === 1 ? 'success' : 'error'}
//                               size="small"
//                               variant="outlined"
//                             />
//                           </TableCell>
//                           <TableCell>
//                             <Chip 
//                               label={result.experience_match.status}
//                               color={getScoreColor(result.experience_match.score)}
//                               size="small"
//                               variant="outlined"
//                             />
//                           </TableCell>
//                           <TableCell>
//                             <Chip 
//                               label={result.education_match.status}
//                               color={getScoreColor(result.education_match.score)}
//                               size="small"
//                               variant="outlined"
//                             />
//                           </TableCell>
//                           <TableCell>
//                             <Chip 
//                               label={result.skills_match.status}
//                               color={getScoreColor(result.skills_match.score)}
//                               size="small"
//                               variant="outlined"
//                             />
//                           </TableCell>
//                           <TableCell>
//                             <IconButton size="small" onClick={() => toggleRow(index)}>
//                               {expandedRows[index] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//                             </IconButton>
//                           </TableCell>
//                         </TableRow>
//                         <TableRow>
//                           <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
//                             <Collapse in={expandedRows[index]} timeout="auto" unmountOnExit>
//                               <Box sx={{ margin: 1, my: 2 }}>
//                                 <Typography variant="h6" gutterBottom component="div">
//                                   Detailed Breakdown
//                                 </Typography>
//                                 <Table size="small">
//                                   <TableHead>
//                                     <TableRow>
//                                       <TableCell>Category</TableCell>
//                                       <TableCell>Details</TableCell>
//                                       <TableCell>Score</TableCell>
//                                     </TableRow>
//                                   </TableHead>
//                                   <TableBody>
//                                     <TableRow>
//                                       <TableCell><strong>Region</strong></TableCell>
//                                       <TableCell>{result.region_match.details}</TableCell>
//                                       <TableCell>{(result.region_match.score * 100).toFixed(0)}%</TableCell>
//                                     </TableRow>
//                                     <TableRow>
//                                       <TableCell><strong>Experience</strong></TableCell>
//                                       <TableCell>{result.experience_match.details}</TableCell>
//                                       <TableCell>{(result.experience_match.score * 100).toFixed(0)}%</TableCell>
//                                     </TableRow>
//                                     <TableRow>
//                                       <TableCell><strong>Education</strong></TableCell>
//                                       <TableCell>{result.education_match.details}</TableCell>
//                                       <TableCell>{(result.education_match.score * 100).toFixed(0)}%</TableCell>
//                                     </TableRow>
//                                     <TableRow>
//                                       <TableCell><strong>Skills</strong></TableCell>
//                                       <TableCell>{result.skills_match.details}</TableCell>
//                                       <TableCell>{(result.skills_match.score * 100).toFixed(0)}%</TableCell>
//                                     </TableRow>
//                                   </TableBody>
//                                 </Table>
//                               </Box>
//                             </Collapse>
//                           </TableCell>
//                         </TableRow>
//                       </React.Fragment>
//                     ))}
//                   </TableBody>
//                 </Table>
//               </TableContainer>
//             </Paper>
//           </Grid>
//         )}
//       </Grid>
//     </Container>
//   );
// }

// export default App;


import { useState, useEffect } from 'react';
import { Upload, File, CheckCircle, XCircle, AlertCircle, Clock, FileText, Briefcase, Award, MapPin, Layers, ChevronDown, ChevronUp } from 'lucide-react';

export default function CVMatcher() {
  const [activeTab, setActiveTab] = useState('job');
  const [jobDescription, setJobDescription] = useState('');
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobRequirements, setJobRequirements] = useState(null);
  const [processingJob, setProcessingJob] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [matchResults, setMatchResults] = useState([]);
  const [expandedResult, setExpandedResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  
  // Process job description
  const processJob = async () => {
    if (!jobDescription.trim()) {
      setErrorMessage('Please enter a job description first.');
      return;
    }

    setProcessingJob(true);
    setErrorMessage('');

    try {
      const response = await fetch('http://localhost:5000/process_job', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job_text: jobDescription }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process job description');
      }

      setJobRequirements(data);
      setCurrentJobId(data.id);
      setActiveTab('cv');
    } catch (error) {
      setErrorMessage(`Error: ${error.message}`);
      console.error('Error processing job:', error);
    } finally {
      setProcessingJob(false);
    }
  };
  
  const EXPERIENCE_KEYWORDS = {
    english: [
      'experience', 'professional experience', 'work experience', 'employment history',
      'career history', 'job history', 'years', 'yrs', 'year', 'yr'
    ],
    french: [
      'expérience', 'expérience professionnelle', 'parcours professionnel', 'historique d\'emploi',
      'ans', 'années', 'année', 'expérience :', 'experience :', 'experience:', 'expérience:'
    ],
    arabic: [
      'خبرة', 'خبرة مهنية', 'تاريخ العمل', 'سنوات', 'سنة'
    ],
  };
  
  // Keywords for skills detection
  const SKILLS_KEYWORDS = {
    english: [
      'skills', 'technical skills', 'competencies', 'expertise', 'proficiencies',
      'abilities', 'qualifications', 'competences', 'technologies'
    ],
    french: [
      'compétences', 'compétences techniques', 'expertises', 'savoir-faire',
      'qualifications', 'technologies', 'outils', 'connaissances', 'maîtrise'
    ],
    arabic: [
      'مهارات', 'مهارات تقنية', 'كفاءات', 'خبرات', 'إتقان'
    ],
  };
  
  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  // Upload and process CVs
  const uploadCVs = async () => {
    if (!files.length) {
      setErrorMessage('Please select CV files first.');
      return;
    }

    if (!currentJobId) {
      setErrorMessage('Please process a job description first.');
      return;
    }

    setUploading(true);
    setErrorMessage('');

    const formData = new FormData();
    formData.append('job_id', currentJobId);
    
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/upload_cvs', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to upload CVs');
      }

      setMatchResults(data.results);
      setActiveTab('results');
    } catch (error) {
      setErrorMessage(`Error: ${error.message}`);
      console.error('Error uploading CVs:', error);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // Toggle expanded view for a result
  const toggleExpand = (index) => {
    if (expandedResult === index) {
      setExpandedResult(null);
    } else {
      setExpandedResult(index);
    }
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'Match':
      case 'Approved':
      case 'Strong match':
        return 'text-green-600';
      case 'Nearly sufficient':
      case 'Good match':
        return 'text-green-600';
      case 'Partially sufficient':
      case 'Moderate match':
        return 'text-yellow-600';
      default:
        return 'text-red-600';
    }
  };

  // Get score color
  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50 pl-[220px]">
    <div className="min-h-screen bg-gray-50">
      {/* <header className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 shadow-md">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold flex items-center">
            <Briefcase className="mr-2" />
            CV-Job Matcher
          </h1>
          <p className="mt-2 text-green-100">Match the best candidates to your job postings automatically</p>
        </div>
      </header> */}
      
      <main className="max-w-6xl mx-auto p-6">
        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-6">
          <button
            className={`py-3 px-6 font-medium ${activeTab === 'job' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('job')}
          >
            1. Job Description
          </button>
          <button
            className={`py-3 px-6 font-medium ${activeTab === 'cv' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500'} ${!jobRequirements ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => jobRequirements && setActiveTab('cv')}
            disabled={!jobRequirements}
          >
            2. Upload CVs
          </button>
          <button
            className={`py-3 px-6 font-medium ${activeTab === 'results' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500'} ${matchResults.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => matchResults.length > 0 && setActiveTab('results')}
            disabled={matchResults.length === 0}
          >
            3. Match Results
          </button>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
            <div className="flex items-center">
              <div className="flex-shrink-0 text-red-500">
                <AlertCircle size={24} />
              </div>
              <div className="ml-3">
                <p className="text-red-700">{errorMessage}</p>
              </div>
            </div>
          </div>
        )}

        {/* Job Description Section */}
        {activeTab === 'job' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <FileText className="mr-2" />
              Enter Job Description
            </h2>
            <p className="text-gray-600 mb-4">
              Paste the complete job description including "Mission" and "Profil" sections. The system will automatically extract requirements.
            </p>
            <textarea
              className="w-full h-64 p-4 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
              placeholder="Paste job description here..."
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
            ></textarea>
            <button
              className={`mt-4 py-2 px-6 bg-green-600 hover:bg-green-700 text-white rounded-md flex items-center ${processingJob ? 'opacity-70 cursor-not-allowed' : ''}`}
              onClick={processJob}
              disabled={processingJob}
            >
              {processingJob ? (
                <>
                  <Clock className="animate-spin mr-2" size={20} />
                  Processing...
                </>
              ) : (
                <>Process Job Description</>
              )}
            </button>
          </div>
        )}

        {/* Job Requirements Display */}
        {jobRequirements && activeTab === 'cv' && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Layers className="mr-2" />
              Extracted Job Requirements
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border rounded-md p-4">
                <h3 className="font-medium text-gray-700 mb-2">Required Education Level</h3>
                <p>{jobRequirements.education_level > 0 ? 
                  ['High School', 'Associate', 'Bachelor\'s', 'Master\'s', 'Doctorate'][jobRequirements.education_level - 1] :
                  'Not specified'}
                </p>
              </div>
              <div className="border rounded-md p-4">
                <h3 className="font-medium text-gray-700 mb-2">Required Experience</h3>
                <p>{jobRequirements.experience > 0 ? `${jobRequirements.experience} years` : 'Not specified'}</p>
              </div>
              <div className="border rounded-md p-4">
                <h3 className="font-medium text-gray-700 mb-2">Region</h3>
                <p>{jobRequirements.region || 'Not specified'}</p>
              </div>
              <div className="border rounded-md p-4">
                <h3 className="font-medium text-gray-700 mb-2">Required Skills</h3>
                {jobRequirements.skills && jobRequirements.skills.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {jobRequirements.skills.map((skill, index) => (
                      <span key={index} className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                        {skill}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p>No specific skills extracted</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* CV Upload Section */}
        {activeTab === 'cv' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Upload className="mr-2" />
              Upload CVs
            </h2>
            <p className="text-gray-600 mb-4">
              Select multiple CV files to match against the job requirements. Supported formats: PDF, DOCX, DOC, TXT.
            </p>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <input
                type="file"
                id="cv-upload"
                multiple
                className="hidden"
                onChange={handleFileChange}
                accept=".pdf,.docx,.doc,.txt"
              />
              <label htmlFor="cv-upload" className="cursor-pointer block">
                <File size={48} className="mx-auto text-gray-400 mb-4" />
                <span className="block text-sm font-medium text-gray-700 mb-1">
                  Click to select files or drag and drop
                </span>
                <span className="block text-xs text-gray-500">
                  PDF, DOCX, DOC, TXT up to 16MB
                </span>
              </label>
            </div>

            {files.length > 0 && (
              <div className="mt-4">
                <h3 className="font-medium text-gray-700 mb-2">Selected Files ({files.length})</h3>
                <ul className="border rounded-md divide-y">
                  {files.map((file, index) => (
                    <li key={index} className="flex items-center p-3">
                      <File size={18} className="text-gray-500 mr-2" />
                      <span className="text-sm">{file.name}</span>
                      <span className="text-xs text-gray-500 ml-2">({(file.size / 1024).toFixed(1)} KB)</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button
              className={`mt-6 py-2 px-6 bg-green-600 hover:bg-green-700 text-white rounded-md flex items-center ${uploading || files.length === 0 ? 'opacity-70 cursor-not-allowed' : ''}`}
              onClick={uploadCVs}
              disabled={uploading || files.length === 0}
            >
              {uploading ? (
                <>
                  <Clock className="animate-spin mr-2" size={20} />
                  Processing CVs...
                </>
              ) : (
                <>Process and Match CVs</>
              )}
            </button>
          </div>
        )}

        {/* Results Section */}
        {activeTab === 'results' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <Award className="mr-2" />
              CV Match Results
            </h2>

            {matchResults.length === 0 ? (
              <div className="text-center py-12">
                <AlertCircle size={48} className="mx-auto text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-700">No results found</h3>
                <p className="text-gray-500">Try uploading different CVs or modifying your job description.</p>
              </div>
            ) : (
              <>
                <div className="mb-6 bg-green-50 p-4 rounded-md">
                  <p className="text-green-700 flex items-center">
                    <AlertCircle size={20} className="mr-2" />
                    Found {matchResults.length} candidate matches. Ranked by match score.
                  </p>
                </div>

                {matchResults.map((result, index) => (
                  <div key={index} className="mb-4 border rounded-lg overflow-hidden">
                    <div 
                      className={`p-4 flex items-center justify-between cursor-pointer ${expandedResult === index ? 'bg-gray-50' : 'bg-white'}`}
                      onClick={() => toggleExpand(index)}
                    >
                      <div className="flex items-center">
                        <div className={`text-2xl font-bold ${getScoreColor(result.overall_score)}`}>
                          {Math.round(result.overall_score * 100)}%
                        </div>
                        <div className="ml-4">
                          <h3 className="font-medium">{result.name}</h3>
                          <div className="text-xs text-gray-500 mt-1">
                            ID: {result.cv_id.substr(0, 8)}...
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center">
                        {/* Quick status indicators */}
                        <div className="hidden md:flex mr-4 space-x-2">
                          <span className={`px-2 py-1 rounded-full text-xs ${result.region_match.score >= 0.8 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                            Region
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs ${result.experience_match.score >= 0.8 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                            Experience
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs ${result.education_match.score >= 0.8 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                            Education
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs ${result.skills_match.score >= 0.8 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                            Skills
                          </span>
                        </div>
                        
                        {expandedResult === index ? (
                          <ChevronUp size={20} className="text-gray-500" />
                        ) : (
                          <ChevronDown size={20} className="text-gray-500" />
                        )}
                      </div>
                    </div>
                    
                    {expandedResult === index && (
                      <div className="p-4 border-t bg-gray-50">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                          {/* Region Match */}
                          <div className="border rounded-md p-3 bg-white">
                            <div className="flex justify-between items-center mb-2">
                              <h4 className="font-medium flex items-center">
                                <MapPin size={16} className="mr-1" />
                                Region Match
                              </h4>
                              <span className={`${getStatusColor(result.region_match.status)}`}>
                                {result.region_match.status}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{result.region_match.details}</p>
                            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${result.region_match.score >= 0.8 ? 'bg-green-500' : result.region_match.score >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                style={{ width: `${result.region_match.score * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          
                          {/* Experience Match */}
                          <div className="border rounded-md p-3 bg-white">
                            <div className="flex justify-between items-center mb-2">
                              <h4 className="font-medium flex items-center">
                                <Briefcase size={16} className="mr-1" />
                                Experience Match
                              </h4>
                              <span className={`${getStatusColor(result.experience_match.status)}`}>
                                {result.experience_match.status}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{result.experience_match.details}</p>
                            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${result.experience_match.score >= 0.8 ? 'bg-green-500' : result.experience_match.score >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                style={{ width: `${result.experience_match.score * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          
                          {/* Education Match */}
                          <div className="border rounded-md p-3 bg-white">
                            <div className="flex justify-between items-center mb-2">
                              <h4 className="font-medium flex items-center">
                                <Award size={16} className="mr-1" />
                                Education Match
                              </h4>
                              <span className={`${getStatusColor(result.education_match.status)}`}>
                                {result.education_match.status}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{result.education_match.details}</p>
                            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${result.education_match.score >= 0.8 ? 'bg-green-500' : result.education_match.score >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                style={{ width: `${result.education_match.score * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          
                          {/* Skills Match */}
                          <div className="border rounded-md p-3 bg-white">
                            <div className="flex justify-between items-center mb-2">
                              <h4 className="font-medium flex items-center">
                                <Layers size={16} className="mr-1" />
                                Skills Match
                              </h4>
                              <span className={`${getStatusColor(result.skills_match.status)}`}>
                                {result.skills_match.status}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600">{result.skills_match.details}</p>
                            <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${result.skills_match.score >= 0.8 ? 'bg-green-500' : result.skills_match.score >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                style={{ width: `${result.skills_match.score * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </main>

      {/* <footer className="bg-gray-800 text-white p-6 mt-12">
        <div className="max-w-6xl mx-auto text-center">
          <p>CV-Job Matcher - Intelligent Candidate Selection System</p>
          <p className="text-gray-400 text-sm mt-2">© 2025 - All rights reserved</p>
        </div>
      </footer> */}
    </div>
    </div>
  );
}
