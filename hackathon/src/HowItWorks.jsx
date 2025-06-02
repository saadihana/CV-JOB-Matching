import React from "react";

function HowItWorks() {
  return (
    <section className="py-12 bg-gray-50">
      <div className="max-w-6xl mx-auto px-4">
        {/* Section Title */}
        <h2 className="text-2xl font-bold text-gray-800 mb-6">How It Works</h2>

        {/* Steps Container (with connecting line in background) */}
        <div className="relative flex flex-col md:flex-row items-center justify-between mt-10">
          {/* Horizontal line behind the steps */}
          <div className="absolute hidden md:block top-1/2 left-0 w-full h-0.5 bg-gray-300" />

          {/* Step 1 */}
          <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
            <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white border-2 border-orange-500 text-orange-500 mb-4 shadow">
              <i className="fas fa-upload text-2xl"></i>
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Upload Files
            </h3>
            <p className="text-gray-600">
              Upload your CSV files containing business registration data
            </p>
          </div>

          {/* Step 2 */}
          <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
            <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white border-2 border-orange-500 text-orange-500 mb-4 shadow">
              <i className="fas fa-robot text-2xl"></i>
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              AI Processing
            </h3>
            <p className="text-gray-600">
              Our AI algorithms analyze and identify redundancies
            </p>
          </div>

          {/* Step 3 */}
          <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4 mb-8 md:mb-0">
            <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white border-2 border-orange-500 text-orange-500 mb-4 shadow">
              <i className="fas fa-check-circle text-2xl"></i>
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Review Results
            </h3>
            <p className="text-gray-600">
              View the processed data with redundancies removed
            </p>
          </div>

          {/* Step 4 */}
          <div className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 px-4">
            <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white border-2 border-orange-500 text-orange-500 mb-4 shadow">
              <i className="fas fa-download text-2xl"></i>
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Export Clean Data
            </h3>
            <p className="text-gray-600">
              Download the filtered results for immediate use
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;
