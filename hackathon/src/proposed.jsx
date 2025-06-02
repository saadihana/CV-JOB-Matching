import React, { useState } from 'react';
import { Check, X, Search, Briefcase } from 'lucide-react';

const ProposedJobs = () => {
  const [showAcceptDialog, setShowAcceptDialog] = useState(false);
  const [showRejectDialog, setShowRejectDialog] = useState(false);
  const [selectedJob, setSelectedJob] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const [jobs, setJobs] = useState([
    {
      id: 1,
      title: "Senior Frontend Developer",
      company: "TechCorp Inc.",
      description: "Looking for an experienced developer to lead our frontend team and implement new features using React and TypeScript.",
      location: "Algiers",
    },
    {
      id: 2,
      title: "UX/UI Designer",
      company: "DesignHub",
      description: "Seeking a creative designer to join our team and help create beautiful, user-friendly interfaces for our clients.",
      location: "Oran",
    },
    {
      id: 3,
      title: "DevOps Engineer",
      company: "CloudScale Solutions",
      description: "Join our DevOps team to help build and maintain our cloud infrastructure using AWS and Kubernetes.",
      location: "Constantine",
    }
  ]);

  const handleAccept = (job) => {
    if (window.confirm('Are you sure you want to accept this job proposal? This action cannot be undone.')) {
      setJobs(jobs.filter(j => j.id !== job.id));
    }
  };

  const handleReject = (job) => {
    if (window.confirm('Are you sure you want to reject this job proposal? This action cannot be undone.')) {
      setJobs(jobs.filter(j => j.id !== job.id));
    }
  };

  const filteredJobs = jobs.filter(job => 
    job.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    job.company.toLowerCase().includes(searchQuery.toLowerCase()) ||
    job.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6 ml-[256px]">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Briefcase className="w-8 h-8 text-orange-500" />
          Proposed Jobs
        </h1>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search jobs..."
              className="pl-10 pr-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 dark:bg-gray-800 focus:ring-2 focus:ring-orange-500 outline-none w-64"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Total Jobs Card */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <div className="flex items-center justify-between pb-2">
          <span className="text-sm font-medium">Total Proposed Jobs</span>
          <Briefcase className="w-4 h-4 text-orange-500" />
        </div>
        <div className="text-2xl font-bold">{jobs.length}</div>
        <p className="text-xs text-gray-500 dark:text-gray-400">Active proposals</p>
      </div>

      {/* Jobs List */}
      <div className="space-y-4">
        {filteredJobs.map(job => (
          <div key={job.id} className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
            <div className="p-6">
              <div className="flex justify-between items-start">
                <div className="space-y-1">
                  <h3 className="text-xl font-semibold">{job.title}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{job.company} • {job.location}</p>
                </div>
              </div>
              <p className="mt-4 text-gray-600 dark:text-gray-300">{job.description}</p>
              <div className="mt-6 flex justify-end">
                <div className="flex gap-3">
                  <button
                    onClick={() => handleReject(job)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 transition-colors"
                  >
                    <X className="w-4 h-4" />
                    Reject
                  </button>
                  <button
                    onClick={() => handleAccept(job)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-100 text-green-600 hover:bg-green-200 dark:bg-green-900/30 dark:hover:bg-green-900/50 transition-colors"
                  >
                    <Check className="w-4 h-4" />
                    Accept
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProposedJobs;
