// // src/InteractiveCrimeBarChart.jsx
// import React, { useRef, useEffect, useMemo, useState } from 'react';
// import * as d3 from 'd3';

// export default function InteractiveCrimeBarChart({ data, startDate, endDate }) {
//   const svgRef     = useRef();
//   const tooltipRef = useRef();

//   // ── CHART DIMENSIONS ──
//   const width   = 1000;
//   const height  = 500;
//   const margin  = { top: 40, right: 20, bottom: 150, left: 100 };

//   // ── CATEGORY SELECTION ──
//   const allCategories = useMemo(
//     () => Array.from(new Set(data.map(d => d.category))).sort(),
//     [data]
//   );
//   const [selectedCats, setSelectedCats] = useState(allCategories);
//   useEffect(() => setSelectedCats(allCategories), [allCategories]);

//   // ── FILTER & AGGREGATE ──
//   const aggregated = useMemo(() => {
//     const filtered = data.filter(d =>
//       (!startDate || d.date >= startDate) &&
//       (!endDate   || d.date <= endDate) &&
//       selectedCats.includes(d.category)
//     );
//     const counts = d3.rollup(filtered, v => v.length, d => d.category);
//     return Array.from(counts, ([category, count]) => ({ category, count }))
//                 .sort((a, b) => b.count - a.count);
//   }, [data, startDate, endDate, selectedCats]);

//   // ── DRAW ──
//   useEffect(() => {
//     const svg     = d3.select(svgRef.current);
//     const tooltip = d3.select(tooltipRef.current);
//     svg.selectAll('*').remove();

//     // scales
//     const x = d3.scaleBand()
//       .domain(aggregated.map(d => d.category))
//       .range([margin.left, width - margin.right])
//       .padding(0.2);

//     const y = d3.scaleLinear()
//       .domain([0, d3.max(aggregated, d => d.count) || 0])
//       .nice()
//       .range([height - margin.bottom, margin.top]);

//     // axes
//     svg.append('g')
//       .attr('transform', `translate(0,${height - margin.bottom})`)
//       .call(d3.axisBottom(x))
//       .selectAll('text')
//         .attr('transform', 'rotate(-45)')
//         .style('text-anchor', 'end');

//     svg.append('g')
//       .attr('transform', `translate(${margin.left},0)`)
//       .call(d3.axisLeft(y));

//     // bars + tooltip
//     svg.selectAll('.bar')
//       .data(aggregated)
//       .join('rect')
//         .attr('class', 'bar')
//         .attr('x',      d => x(d.category))
//         .attr('y',      d => y(d.count))
//         .attr('width',  x.bandwidth())
//         .attr('height', d => height - margin.bottom - y(d.count))
//         .on('mouseover', (event, d) => {
//           const rect = svgRef.current.getBoundingClientRect();
//           tooltip
//             .style('opacity', 1)
//             .html(`<strong>${d.category}</strong><br/>Count: ${d.count}`)
//             .style('left',  (event.clientX - rect.left + 10) + 'px')
//             .style('top',   (event.clientY - rect.top + 10) + 'px');
//         })
//         .on('mousemove', event => {
//           const rect = svgRef.current.getBoundingClientRect();
//           tooltip
//             .style('left', (event.clientX - rect.left + 10) + 'px')
//             .style('top',  (event.clientY - rect.top + 10) + 'px');
//         })
//         .on('mouseout', () => {
//           tooltip.style('opacity', 0);
//         });
//   }, [aggregated]);

//   // ── CATEGORY TOGGLE ──
//   const toggleCategory = (cat) =>
//     setSelectedCats(prev =>
//       prev.includes(cat)
//         ? prev.filter(c => c !== cat)
//         : [...prev, cat]
//     );

//   // ── TITLE FORMATTING ──
//   const fmt = d3.timeFormat('%Y-%m-%d');
//   const titleText = startDate && endDate
//     ? `Crime Counts from ${fmt(startDate)} to ${fmt(endDate)}`
//     : 'Crime Counts';

//   // ── RENDER ──
//   return (
//     <div>
//       {/* Title */}
//       <h3 className="text-xl font-semibold mb-4">{titleText}</h3>

//       <div className="flex">
//         {/* Checkbox List */}
//         <div
//           className="w-48 overflow-auto pr-4"
//           style={{ maxHeight: height }}
//         >
//           {allCategories.map(cat => (
//             <label key={cat} className="block text-sm mb-1">
//               <input
//                 type="checkbox"
//                 className="mr-2"
//                 checked={selectedCats.includes(cat)}
//                 onChange={() => toggleCategory(cat)}
//               />
//               {cat}
//             </label>
//           ))}
//         </div>

//         {/* SVG + Tooltip */}
//         <div className="flex-1 relative">
//           <svg
//             ref={svgRef}
//             viewBox={`0 0 ${width} ${height}`}
//             preserveAspectRatio="xMinYMin meet"
//             style={{ width: '100%', height: '100%' }}
//           />
//           <div
//             ref={tooltipRef}
//             style={{
//               position:      'absolute',
//               opacity:       0,
//               pointerEvents: 'none',
//               backgroundColor: '#fff',
//               border:        '1px solid #ccc',
//               padding:       '6px',
//               borderRadius:  '4px',
//               fontSize:      '12px',
//               boxShadow:     '0px 0px 6px rgba(0,0,0,0.1)'
//             }}
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// src/InteractiveCrimeBarChart.jsx
import React, { useRef, useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';

const API_BASE_URL = 'http://localhost:8001';

export default function InteractiveCrimeBarChart({ endDate }) {
  const svgRef = useRef();
  const tooltipRef = useRef();
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ── FETCH DATA ──
  useEffect(() => {
    const fetchData = async () => {
      if (!endDate) return;
      
      setLoading(true);
      setError(null);
      
      try {
        // Format the date to YYYY-MM-DD
        const formattedDate = new Date(endDate).toISOString().split('T')[0];
        
        const response = await fetch(
          `${API_BASE_URL}/api/crime-data?end_date=${formattedDate}`,
          {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }
          }
        );
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        const jsonData = await response.json();
        if (!Array.isArray(jsonData)) {
          throw new Error('Invalid data format received from server');
        }

        if (jsonData.length === 0) {
          setData([]);
          return;
        }
        
        // Process and validate the data
        const validData = jsonData
          .filter(item => item.incident_datetime && item.incident_category)
          .map(item => {
            try {
              const date = new Date(item.incident_datetime);
              if (isNaN(date.getTime())) {
                return null;
              }
              return {
                date,
                category: item.incident_category || 'Unknown'
              };
            } catch (err) {
              console.warn('Invalid date format:', item.incident_datetime);
              return null;
            }
          })
          .filter(item => item !== null);

        setData(validData);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching crime data:', err);
        setData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [endDate]);

  // ── CHART DIMENSIONS ──
  const width = 1000;
  const height = 600;
  const margin = { top: 40, right: 20, bottom: 50, left: 200 };

  // ── CATEGORY SELECTION ──
  const allCategories = useMemo(
    () => Array.from(new Set(data.map(d => d.category))).sort(),
    [data]
  );
  const [selectedCats, setSelectedCats] = useState(allCategories);
  useEffect(() => setSelectedCats(allCategories), [allCategories]);

  // ── FILTER & AGGREGATE ──
  const aggregated = useMemo(() => {
    const filtered = data.filter(d => selectedCats.includes(d.category));
    const counts = d3.rollup(filtered, v => v.length, d => d.category);
    return Array.from(counts, ([category, count]) => ({ category, count }))
                .sort((a, b) => b.count - a.count);
  }, [data, selectedCats]);

  // ── DRAW (horizontal bars with truncated labels) ──
  useEffect(() => {
    if (!aggregated.length) return;

    const svg = d3.select(svgRef.current);
    const tooltip = d3.select(tooltipRef.current);
    svg.selectAll('*').remove();

    // 1) x scale: linear from 0 → max count
    const x = d3.scaleLinear()
      .domain([0, d3.max(aggregated, d => d.count) || 0])
      .nice()
      .range([margin.left, width - margin.right]);

    // 2) y scale: band for categories
    const y = d3.scaleBand()
      .domain(aggregated.map(d => d.category))
      .range([margin.top, height - margin.bottom])
      .padding(0.2);

    // 3) Bottom x-axis (counts)
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
        .style('text-anchor', 'middle');

    // 4) Left y-axis (category labels, truncated + tooltip)
    const yAxisG = svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));

    // For each tick, truncate to first 12 chars and add title element for hover
    const maxLabelLen = 12;
    yAxisG.selectAll('text')
      .each(function(d) {
        const element = d3.select(this);
        const fullText = d;
        const truncated = fullText.length > maxLabelLen
                         ? fullText.slice(0, maxLabelLen) + '...'
                         : fullText;
        
        // Set the truncated text
        element.text(truncated);
        
        // Add title element for hover tooltip
        element.append('title').text(fullText);
      });

    // 5) Bars (horizontal)
    svg.selectAll('.bar')
      .data(aggregated)
      .join('rect')
        .attr('class', 'bar')
        .attr('x', x(0))
        .attr('y', d => y(d.category))
        .attr('height', y.bandwidth())
        .attr('width', d => x(d.count) - x(0))
        .attr('fill', '#69b3a2')
        .on('mouseover', (event, d) => {
          const rect = svgRef.current.getBoundingClientRect();
          tooltip
            .style('opacity', 1)
            .html(`<strong>${d.category}</strong><br/>Count: ${d.count}`)
          .style('left', (event.clientX - rect.left + 10) + 'px')
          .style('top', (event.clientY - rect.top + 10) + 'px');
        })
        .on('mousemove', event => {
          const rect = svgRef.current.getBoundingClientRect();
          tooltip
            .style('left', (event.clientX - rect.left + 10) + 'px')
          .style('top', (event.clientY - rect.top + 10) + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('opacity', 0);
        });
  }, [aggregated]);

  // ── CATEGORY TOGGLE ──
  const toggleCategory = (cat) =>
    setSelectedCats(prev =>
      prev.includes(cat)
        ? prev.filter(c => c !== cat)
        : [...prev, cat]
    );

  // ── RENDER ──
  if (loading) {
    return <div>Loading crime data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!data.length) {
    return <div>No crime data available for the selected date range.</div>;
  }

  return (
    <div>
      <div className="flex">
        {/* Checkbox List */}
        <div
          className="w-56 overflow-auto pr-4"
          style={{ maxHeight: height }}
        >
          {allCategories.map(cat => (
            <label key={cat} className="block text-sm mb-1">
              <input
                type="checkbox"
                className="mr-2"
                checked={selectedCats.includes(cat)}
                onChange={() => toggleCategory(cat)}
              />
              {cat}
            </label>
          ))}
        </div>

        {/* SVG + Tooltip */}
        <div className="flex-1 relative">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMinYMin meet"
            style={{ width: '100%', height: '100%' }}
          />
          <div
            ref={tooltipRef}
            style={{
              position: 'absolute',
              opacity: 0,
              pointerEvents: 'none',
              backgroundColor: '#fff',
              border: '1px solid #ccc',
              padding: '6px',
              borderRadius: '4px',
              fontSize: '12px',
              boxShadow: '0px 0px 6px rgba(0,0,0,0.1)',
            }}
          />
        </div>
      </div>
    </div>
  );
}
