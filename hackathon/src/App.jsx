




import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "./About"; // Assuming About.jsx is your dashboard
import Results from "./Histo";   // Histo.jsx for results
import Navbar from "./Navbar";
import History from "./history";
import Proposed from "./proposed";


const App = () => {
  return (
    <div>
      <Navbar />
      <Routes>
        
        <Route path="/" element={<Navigate replace to="/dashboard" />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/results" element={<Results />} />
        <Route path="/history" element={<History />} />
        <Route path="/proposed" element={<Proposed />} />
      </Routes>
    </div>
  );
};

export default App;
