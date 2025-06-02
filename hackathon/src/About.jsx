import React, { useState, useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { 
  Filter, Home, Upload, History, BarChart2, Settings, 
  Moon, Sun, Play, Rocket, Check, Bot, FileText 
} from 'lucide-react';
import mobilisLogo from './assets/mobilis.jpg'

// Particle component (unchanged)
const Particle = ({ delay }) => {
  const size = Math.random() * 15 + 5;
  return (
    <motion.div
      initial={{ 
        opacity: 0.1,
        x: `${Math.random() * 100}%`,
        y: `${Math.random() * 100}%`
      }}
      animate={{
        y: ['0%', '100%'],
        opacity: [0.1, 0.5, 0.1]
      }}
      transition={{
        duration: Math.random() * 10 + 10,
        delay,
        repeat: Infinity,
        ease: "linear"
      }}
      className="absolute rounded-full bg-white/10"
      style={{
        width: size,
        height: size
      }}
    />
  );
};

const About = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [activeMenu, setActiveMenu] = useState('dashboard');
  const [uploadedFiles, setUploadedFiles] = useState([
    { id: 1, name: 'business_data_2024.csv', size: '3.2 MB', progress: 100, status: 'success' },
    { id: 2, name: 'registration_batch_feb.csv', size: '1.8 MB', progress: 100, status: 'success' },
  ]);

  // Create a ref for the "How It Works" section
  const howItWorksRef = useRef(null);

  // Function to scroll to the "How It Works" section
  const scrollToHowItWorks = () => {
    if (howItWorksRef.current) {
      howItWorksRef.current.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' // Ensures the section is centered in the viewport
      });
    }
  };

  return (
    <div className={`flex h-screen overflow-hidden ${darkMode ? 'dark' : ''}`}>
      {/* Main Content */}
      <div className="flex-1 ml-64 overflow-y-auto bg-gray-50 dark:bg-gray-900">
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="relative bg-gradient-to-r bg-gray-50 to-indigo-900 text-gray text-center py-20 px-6 dark:bg-gray-900"
        >
          {/* Particles */}
          <div className="absolute inset-0 overflow-hidden">
            {Array.from({ length: 20 }).map((_, i) => (
              <Particle key={i} delay={i * 0.2} />
            ))}
          </div>

          <div className="relative z-10 max-w-3xl mx-auto">
            {/* Static Logo - No Animation */}
            <div className="mb-4">
              <img 
                src={mobilisLogo}
                alt="CV-Match Logo" 
                className="mx-auto w-16 h-16 object-contain" 
              />
            </div>

            

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-4xl md:text-5xl font-bold mb-4"
            >
              <span className="text-fray dark:text-white">Welcome to </span>
              <span className="text-green-500 ">SmartHire</span>
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-6"
            >
              <p className="text-lg opacity-80 max-w-2xl mx-auto dark:text-white">
                SmartHire is here to guide you match the best CVs with the right job offers.  
              </p>

              <div className="flex flex-wrap justify-center gap-4">
                <motion.button 
                  whileHover={{ scale: 1.05, y: -2 }}
                  className="bg-green-500 text-gray-900 font-bold py-3 px-6 rounded-full flex items-center gap-2"
                  onClick={scrollToHowItWorks} // Scroll to "How It Works" section
                >
                  <Rocket size={18} />
                  Learn More
                </motion.button>
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Description Section */}
        <section ref={howItWorksRef} className="py-12 bg-gray-50 dark:bg-gray-900">
          <div className="max-w-6xl mx-auto px-4">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">
              About SmartHire
            </h2>
            <div className="space-y-6 text-gray-600 dark:text-gray-400">
              <p>
              SmartHire is a smart recruitment solution dedicated to connecting top talent with the right opportunities at Mobilis. Our mission is simple: to streamline the hiring process by intelligently matching candidates’ CVs with job openings that best fit their skills, experience, and aspirations.
              </p>
              <p>
              Powered by advanced algorithms and a deep understanding of Mobilis’ needs, SmartHire ensures that every applicant is seen for what they truly bring to the table. Whether you're looking for your next career move or aiming to build a stronger team, SmartHire helps make the connection seamless, accurate, and efficient.
              </p>
              <p>
              We believe in talent. We believe in opportunity. And most importantly, we believe in making the right match.
              </p>
            </div>
          </div>
        </section>

        {/* How It Works Section (unchanged) */}
        <section className="py-12 bg-gray-50 dark:bg-gray-900">
          <div className="max-w-6xl mx-auto px-4">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-6">How It Works</h2>
            <div className="relative flex flex-col md:flex-row items-center justify-between mt-10">
              {/* Step 1: Upload Files */}
              <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white dark:bg-gray-800 border-2 border-yellow-500 text-yellow-500 mb-4 shadow">
                  <Upload className="text-2xl" />
                </div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Upload the CVs
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Upload the CVs directly. We support PDF and DOCX formats.
                </p>
              </div>

              {/* Step 2: AI Processing */}
              <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white dark:bg-gray-800 border-2 border-red-500 text-red-500 mb-4 shadow">
                  <Bot className="text-2xl" />
                </div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  AI-Powered Matching
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                using models that assess skills, experience, and qualifications.
                </p>
              </div>

              {/* Step 3: Review Results */}
              <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white dark:bg-gray-800 border-2 border-orange-500 text-orange-500 mb-4 shadow">
                  <Check className="text-2xl" />
                </div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                View the Matches 
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                Instantly see the best match, including compatibility score..
                </p>
              </div>

              {/* Step 4: Export Clean Data */}
              <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white dark:bg-gray-800 border-2 border-green-500 text-green-500 mb-4 shadow">
                  <History className="text-2xl" />
                </div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Explore Personalized Insights
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                Statistical details: Match Percentage, top Matching Skills
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer Section (unchanged) */}
        <footer className="bg-white dark:bg-gray-900 py-12 transition-all duration-300">
          <div className="max-w-6xl mx-auto px-4">
            <h2 className="text-2xl font-bold mb-6 text-gray-800 dark:text-gray-100">
              Key Features
            </h2>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              {/* 1. AI-Powered Analysis */}
              <div className="p-6 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm transition-all duration-300 hover:shadow-lg hover:scale-105">
                <Bot className="text-green-500" strokeWidth={2.5} />
                <h3 className="flex items-center gap-2 text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                AI-Powered Job Matching
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                Our advanced AI algorithms match candidates' 
                CVs with job listings based on skills, experience, 
                and career goals, ensuring the best fit for both employers and job seekers.
                </p>
              </div>

              {/* 2. Document Identification */}
              <div className="p-6 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm transition-all duration-300 hover:shadow-lg hover:scale-105">
                <FileText className="text-green-500" strokeWidth={2.5} />
                <h3 className="flex items-center gap-2 text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Real-Time Candidate Ranking
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                CVs are ranked in real time based on their compatibility 
                with job openings, allowing employers to instantly identify 
                the most qualified applicants.
                </p>
              </div>

              {/* 3. Advanced Analytics */}
              <div className="p-6 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm transition-all duration-300 hover:shadow-lg hover:scale-105">
                <BarChart2 className="text-green-500" strokeWidth={2.5} />
                <h3 className="flex items-center gap-2 text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Customizable Matching Criteria
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                Employers can define specific criteria such as skills, 
                experience, and location, ensuring CVs are matched according 
                to unique job requirements.
                </p>
              </div>

              {/* 4. Smart Filtering */}
              <div className="p-6 bg-gray-50 dark:bg-gray-700 rounded-lg shadow-sm transition-all duration-300 hover:shadow-lg hover:scale-105">
                <Filter className="text-green-500" strokeWidth={2.5} />
                <h3 className="flex items-center gap-2 text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Automated CV Parsing
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                Key information from uploaded CVs, including skills, 
                job history, and education, is automatically extracted, 
                saving recruiters time on manual data entry.
                </p>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default About;