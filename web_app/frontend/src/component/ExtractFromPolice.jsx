import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import Papa from 'papaparse';
import 'mapbox-gl/dist/mapbox-gl.css';

export default function HeatMap() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  // State to hold all parsed CSV rows
  const [rows, setRows] = useState([]);

  // Extracted unique categories from the CSV
  const [categories, setCategories] = useState([]);

  // Currently selected category
  const [selectedCategory, setSelectedCategory] = useState('');

  //
  // 1) Load CSV once and extract rows + unique categories
  //
  useEffect(() => {
    Papa.parse(
      '/data/Police_Department_Incident_Reports__2018_to_Present_20250507.csv',
      {
        header: true,
        download: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          const dataRows = results.data;

          // Save all rows into state
          setRows(dataRows);

          // Extract unique categories (sorted)
          const uniqueCats = Array.from(
            new Set(dataRows.map((r) => r['Incident Category']))
          ).sort();
          setCategories(uniqueCats);

          // Default to the first category in the dropdown (if any)
          if (uniqueCats.length > 0) {
            setSelectedCategory(uniqueCats[0]);
          }
        },
        error: (err) => {
          console.error('PapaParse error:', err);
        }
      }
    );
  }, []);

  //
  // 2) Initialize the Mapbox map ONCE
  //
  useEffect(() => {
    mapboxgl.accessToken =
      'pk.eyJ1IjoibWVsaW5kYTAzMjYiLCJhIjoiY21iaGNrdDNrMDd4cjJscHFvcWxwa2NsbCJ9.xXrM3t_PX0ezAbpXwI41Cw';

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v11',
      center: [-122.4194, 37.7749],
      zoom: 12,
      maxZoom: 15
    });

    // Keep a ref to the map instance
    mapRef.current = map;

    map.on('style.load', () => {
      // Add an empty GeoJSON source; we'll populate it when a category is selected
      map.addSource('incidents', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        }
      });

      // Add the heatmap layer (initially empty)
      map.addLayer({
        id: 'incident-heatmap',
        type: 'heatmap',
        source: 'incidents',
        maxzoom: 15,
        paint: {
          // Each point contributes weight = 1
          'heatmap-weight': [
            'interpolate',
            ['linear'],
            ['get', 'point_count'],
            0,
            0,
            1,
            1
          ],
          // Increase intensity as you zoom in
          'heatmap-intensity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0,
            1,
            15,
            3
          ],
          // Color ramp: low density → transparent, high → deep red
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0,
            'rgba(0, 0, 255, 0)',     // fully transparent
            0.2,
            'rgb(103, 169, 207)',
            0.4,
            'rgb(209, 229, 240)',
            0.6,
            'rgb(253, 219, 199)',
            0.8,
            'rgb(239, 138, 98)',
            1,
            'rgb(178, 24, 43)'
          ],
          // Radius of influence grows with zoom
          'heatmap-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0,
            2,
            15,
            20
          ],
          // Optionally fade out the heatmap at very high zooms
          'heatmap-opacity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            7,
            1,
            15,
            0
          ]
        }
      });
    });

    // Clean up on unmount
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  //
  // 3) Re-filter rows & update map source whenever
  //    - rows have finished loading, OR
  //    - selectedCategory changes.
  //
  useEffect(() => {
    const map = mapRef.current;
    if (!map || rows.length === 0 || !selectedCategory) return;

    // Compute “one year ago” threshold
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);

    // Filter rows by:
    //   • matching the selected category
    //   • valid lat/lng
    //   • date ≥ oneYearAgo
    const filtered = rows.filter((row) => {
      if (row['Incident Category'] !== selectedCategory) return false;

      // Ensure latitude/longitude are present and numeric
      if (
        row['Latitude'] == null ||
        row['Longitude'] == null ||
        isNaN(row['Latitude']) ||
        isNaN(row['Longitude'])
      )
        return false;

      // Parse incident date
      const incidentDate = new Date(row['Incident Date']);
      if (isNaN(incidentDate.getTime())) return false;
      if (incidentDate < oneYearAgo) return false;

      return true;
    });

    // Build GeoJSON features from the filtered set
    const features = filtered.map((row) => ({
      type: 'Feature',
      properties: { category: row['Incident Category'] },
      geometry: {
        type: 'Point',
        coordinates: [row['Longitude'], row['Latitude']]
      }
    }));

    const geojsonData = {
      type: 'FeatureCollection',
      features: features
    };

    // Update the “incidents” source data
    if (map.getSource('incidents')) {
      map.getSource('incidents').setData(geojsonData);
    }
  }, [rows, selectedCategory]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* ── Header + Dropdown ── */}
      <div style={{ textAlign: 'center', margin: '10px 0' }}>
        <h2 style={{ fontSize: '1rem', fontWeight: '500' }}>
          {selectedCategory
            ? `${selectedCategory} (Past 12 Months)`
            : 'Select a Category'}
        </h2>

        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          style={{
            padding: '6px 10px',
            fontSize: '1rem',
            borderRadius: '4px',
            border: '1px solid #ccc'
          }}
        >
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </div>

      {/* ── Map Container ── */}
      <div
        ref={mapContainer}
        style={{
          flex: 1,
          width: '100%',
          position: 'relative'
        }}
      />
    </div>
  );
}