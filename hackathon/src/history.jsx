import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Download, PieChart, Trash2, Search, Calendar, Filter, FileText, Ban, AlertCircle } from 'lucide-react';
import Navbar from './Navbar'; // Import your existing Navbar component

const HistoryPage = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = React.useState('');
  const [dateFilter, setDateFilter] = React.useState('All Dates');
  const [statusFilter, setStatusFilter] = React.useState('All Status');

  const fileHistory = [
    // ... (keep your existing fileHistory array)
  ];

  const getStatusClass = (status) => {
    switch (status) {
      case 'Completed':
        return 'bg-blue-100 text-blue-600';
      case 'High Redundancy':
        return 'bg-orange-100 text-orange-600';
      case 'Processing Failed':
        return 'bg-red-100 text-red-600';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  const filteredFiles = fileHistory.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'All Status' || file.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navbar />
      
      <div className="ml-64 p-8 min-h-screen">
        {/* Header Section */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">File History</h1>
          <div className="flex items-center gap-3">
            <span className="font-medium text-gray-700 dark:text-gray-300">Admin User</span>
            <div className="w-10 h-10 bg-blue-900 text-white rounded-full flex items-center justify-center font-bold">
              A
            </div>
          </div>
        </div>

        {/* Search & Filter Section */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border-l-4 border-green-500 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <Search className="w-5 h-5 text-green-500" />
            <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Search & Filter</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search by file name..."
                className="w-full pl-10 pr-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg text-sm bg-transparent dark:text-gray-300"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="relative">
              <select 
                className="w-full pl-4 pr-10 py-2 border border-gray-200 dark:border-gray-700 rounded-lg text-sm appearance-none bg-transparent dark:text-gray-300"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
              >
                <option>All Dates</option>
                <option>Last 7 Days</option>
                <option>Last 30 Days</option>
                <option>Last 90 Days</option>
                <option>Custom Range</option>
              </select>
              <Calendar className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            </div>
            
            <div className="relative">
              <select 
                className="w-full pl-4 pr-10 py-2 border border-gray-200 dark:border-gray-700 rounded-lg text-sm appearance-none bg-transparent dark:text-gray-300"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option>All Status</option>
                <option>High Redundancy</option>
                <option>Completed</option>
                <option>Processing Failed</option>
              </select>
              <Filter className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>

        {/* File History Table */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-green-500" />
              <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Processed Files</h2>
            </div>
            <span className="px-3 py-1 bg-blue-900 text-white text-sm font-medium rounded-full">
              {filteredFiles.length} Files
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  {['File', 'Upload Date', 'Status', 'Result File', 'Actions'].map((header) => (
                    <th 
                      key={header}
                      className="p-6 text-left text-sm font-semibold text-gray-600 dark:text-gray-400"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredFiles.map((file, index) => (
                  <tr 
                    key={index}
                    className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <td className="p-6">
                      <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-green-500" />
                        <div>
                          <div className="font-medium text-gray-900 dark:text-gray-200">{file.name}</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {file.size} • {file.records} records
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="p-6">
                      <div>
                        <div className="font-medium text-gray-900 dark:text-gray-200">{file.uploadDate}</div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">{file.uploadTime}</div>
                      </div>
                    </td>
                    <td className="p-6">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusClass(file.status)}`}>
                        {file.status}
                      </span>
                    </td>
                    <td className="p-6">
                      {file.resultFile ? (
                        <div className="flex items-center gap-3">
                          <FileText className="w-5 h-5 text-green-500" />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-200">{file.resultFile.name}</div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {file.resultFile.size} • {file.resultFile.records} records
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <Ban className="w-5 h-5 text-red-500" />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-200">No Result File</div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">Processing error occurred</div>
                          </div>
                        </div>
                      )}
                    </td>
                    <td className="p-6">
                      <div className="flex gap-2">
                        <button 
                          className={`p-2 rounded-lg ${
                            file.resultFile 
                              ? 'bg-blue-50 text-blue-600 hover:bg-blue-100 dark:bg-blue-900/20 dark:hover:bg-blue-900/30' 
                              : 'bg-gray-50 text-gray-400 cursor-not-allowed dark:bg-gray-700'
                          }`}
                          disabled={!file.resultFile}
                          aria-label="Download file"
                        >
                          <Download className="w-5 h-5" />
                        </button>
                        <button 
                          className={`p-2 rounded-lg ${
                            file.status === 'Processing Failed'
                              ? 'bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/30'
                              : 'bg-green-50 text-green-600 hover:bg-green-100 dark:bg-green-900/20 dark:hover:bg-green-900/30'
                          }`}
                          aria-label={file.status === 'Processing Failed' ? 'View error' : 'View analytics'}
                        >
                          {file.status === 'Processing Failed' ? (
                            <AlertCircle className="w-5 h-5" />
                          ) : (
                            <PieChart className="w-5 h-5" />
                          )}
                        </button>
                        <button 
                          className="p-2 rounded-lg bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/30"
                          aria-label="Delete file"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Showing 1-{filteredFiles.length} of {filteredFiles.length} files
            </span>
            <div className="flex gap-2">
              <button className="p-2 rounded-lg border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700">
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button className="p-2 rounded-lg bg-blue-900 text-white">1</button>
              <button className="p-2 rounded-lg border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700">
                2
              </button>
              <button className="p-2 rounded-lg border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700">
                3
              </button>
              <button className="p-2 rounded-lg border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700">
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryPage;