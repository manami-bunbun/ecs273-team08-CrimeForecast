// src/App.jsx
import React, { useState } from 'react';
import './App.css';

export default function App() {
  // pull the selected date‐range into state
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  return (
    <div className="flex h-screen w-screen">
      
      {/* ── LEFT COLUMN ── */}
      <div className="flex-3 flex flex-col p-4 bg-white">
        {/* Header */}
        <header className="flex items-center mb-6">
          <img src="/logo.png" alt="SFPD Logo" className="w-10 h-10 mr-3" />
          <div>
            <h1 className="text-2xl font-semibold">Crime Data</h1>
            <p className="text-sm text-gray-600">
              Comparing a Date Range Within One Year to Prior Year
            </p>
          </div>
        </header>

        {/* Date Pickers */}
        <div className="mb-6 flex space-x-4">
          <label className="flex flex-col">
            <span className="text-sm mb-1">Start Date</span>
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="border rounded p-2"
            />
          </label>
          <label className="flex flex-col">
            <span className="text-sm mb-1">End Date</span>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="border rounded p-2"
            />
          </label>
        </div>

        {/* Chart */}
        <div className="flex-1 mb-6 border rounded-lg p-4">
          {/* <CrimeChart startDate={startDate} endDate={endDate} /> */}
          <h2 className="text-lg font-medium mb-2">Crime Trend Chart</h2>
        </div>

        {/* Table */}
        <div className="border rounded-lg p-4 overflow-auto">
          {/* <CrimeTable startDate={startDate} endDate={endDate} /> */}
          <h2 className="text-lg font-medium mb-2">Summary Table</h2>
        </div>
      </div>

      {/* ── RIGHT COLUMN ── */}
      <aside className="flex-1 flex flex-col p-4 bg-gray-50">
        {/* Related News */}
        <section className="mb-6 border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">Related News</h2>
          {/* <NewsList /> */}
        </section>

        {/* LLM Advice */}
        <section className="border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">LLM Advice</h2>
          {/* <AdvicePanel /> */}
        </section>
      </aside>
    </div>
  );
}
