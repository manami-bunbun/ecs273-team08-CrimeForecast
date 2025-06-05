// src/App.jsx
import React, { useState, useEffect } from 'react';
import './App.css';
import * as d3 from 'd3';
import InteractiveCrimeBarChart from './component/CrimeBar';
import ExtractHeatMap from './component/ExtractFromPolice';
import 'antd/dist/reset.css';
import NewsList from './component/news_list';
import AdvicePanel from './component/advice_panel';

// Configure API base URL
const API_BASE_URL = 'http://localhost:8001';

export default function App() {
  // Date‐picker state (strings in "YYYY-MM-DD")
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [data, setData] = useState([]);
  const [showBarChart, setShowBarChart] = useState(false);

  useEffect(() => {
    const csvUrl = '/data/Police_Department_Incident_Reports__2018_to_Present_20250507.csv';
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
            <h2 className="text-3xl font-bold mb-1">
              Crime Data
            </h2>
            <p className="text-sm text-gray-600">
              {/* Comparing a Date Range within One Year to Its Prior Year */}
            </p>
          </div>
        </header>

        {/* ── DATE PICKER + TOGGLE BUTTON ROW ── */}
        <div className="flex items-center justify-between mb-6">
        <label className="flex flex-col">
        <span className="text-sm mb-1">End Date (Shows data for previous 30 days)</span>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="border rounded p-2 flex-1 max-w-xs"
            />
          </label>

          <button
            onClick={() => setShowBarChart(!showBarChart)}
            className="ml-4 px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors flex items-center border border-gray-300"
          >
            {showBarChart ? 'Show HeatMap' : 'Show Bar Chart'}
          </button>
        </div>

        {/* ── FLEXIBLE CONTAINER FOR BAR CHART + MAP ── */}
        <div className="flex-1 border rounded-lg overflow-hidden">
          {showBarChart ? (
            <InteractiveCrimeBarChart
              data={data}
              startDate={startDate ? new Date(startDate) : undefined}
              endDate={endDate ? new Date(endDate) : undefined}
            />
          ) : (
            <ExtractHeatMap />
          )}
        </div>
      </div>

      {/* ── RIGHT COLUMN ── */}
      <aside className="flex-1 flex flex-col p-4 bg-white" style={{ 
        maxHeight: 'calc(100vh - 2rem)', // Full height minus padding
        overflowY: 'auto',
        position: 'sticky',
        top: '1rem'
      }}>
        <section className="mb-6 border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">Related News</h2>
          <NewsList endDate={endDate}/>
        </section>
        <section className="border rounded-lg p-4 bg-white">
          <h2 className="text-xl font-semibold mb-3">LLM Advice</h2>
          <AdvicePanel endDate={endDate} />
        </section>
      </aside>
    </div>
  );
}

// export default function App(){
//   return <ExtractHeatMap/>
// }

