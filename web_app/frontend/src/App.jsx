// src/App.jsx
import React, { useState, useEffect } from 'react';
import './App.css';
import * as d3 from 'd3';
import InteractiveCrimeBarChart from './component/CrimeBar';
import ExtractHeatMap from './component/ExtractFromPolice';

export default function App() {
  // Date‐picker state (strings in "YYYY-MM-DD")
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate]   = useState('');
  // Loaded and parsed CSV rows
  const [data, setData] = useState([]);

  useEffect(() => {
    const csvUrl = '/data/Police_Department_Incident_Reports__2018_to_Present_20250507.csv';
    // parse strings like "2023/03/01"
    const parseDate = d3.timeParse('%Y/%m/%d');

    d3.csv(csvUrl, row => ({
      date: parseDate(row['Incident Date']),
      category: row['Incident Category']
    }))
    .then(rows => {
      const clean = rows.filter(d => d.date && d.category);
      setData(clean);
    })
    .catch(err => console.error('CSV load error:', err));
  }, []);

  return (
    /* ── OUTERMOST WRAPPER ──
       Make sure it’s white all the way through.
       We use w-screen/h-screen so it always covers the full viewport. */
    <div className="flex w-screen h-screen bg-white">

      {/* ── LEFT COLUMN ── */}
      <div className="flex-3 flex flex-col p-4 bg-white h-full">
        {/* Header */}
        <header className="flex items-center mb-6">
          <img
            src="/sfpd.png"
            alt="SFPD Logo"
            className="w-16 h-16 mr-4"
          />
          <div>
            <p className="text-xs uppercase font-medium tracking-wide mb-1">
              San Francisco Police Department
            </p>
            <h1 className="text-3xl font-bold mb-1">
              Crime Data
            </h1>
            <p className="text-sm text-gray-600">
              Comparing a Date Range within One Year to Its Prior Year
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

        {/* ── FLEXIBLE CONTAINER FOR BAR CHART + MAP ── */}
        <div className="flex-1 flex flex-col space-y-6">
          {/* Crime Trend Chart (takes half the space) */}
          <div className="flex-1 border rounded-lg p-4 bg-white">
            <InteractiveCrimeBarChart
              data={data}
              startDate={startDate ? new Date(startDate) : undefined}
              endDate={endDate   ? new Date(endDate)   : undefined}
            />
          </div>

          {/* Category Map Box (takes the other half) */}
          <div className="flex-1 border rounded-lg p-4 flex flex-col bg-white">
            <h2 className="text-lg font-medium mb-2">Category Map</h2>
            <div className="flex-1">
              <ExtractHeatMap />
            </div>
          </div>
        </div>
      </div>

      {/* ── RIGHT COLUMN ── */}
      <aside className="flex-1 flex flex-col p-4 bg-white">
        <section className="mb-6 border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">Related News</h2>
          {/* Related News Content Here */}
        </section>
        <section className="border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">LLM Advice</h2>
          {/* LLM Advice Content Here */}
        </section>
      </aside>
    </div>
  );
}

// export default function App(){
//   return <ExtractHeatMap/>
// }

