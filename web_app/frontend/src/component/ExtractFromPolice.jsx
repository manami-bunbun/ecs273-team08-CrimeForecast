import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const API_BASE_URL = 'http://localhost:8001';

export default function HeatMap() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  // State to hold crime data
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Extracted unique categories from the data
  const [categories, setCategories] = useState(['All']);
  const [allCategories, setAllCategories] = useState(['All']);

  // Currently selected category
  const [selectedCategory, setSelectedCategory] = useState('All');

  // Current end date (defaults to today)
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

  // Fetch all categories once when component mounts
  useEffect(() => {
    const fetchAllCategories = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/crime-locations?category=All&end_date=${new Date().toISOString().split('T')[0]}`,
          {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonData = await response.json();
        
        if (!Array.isArray(jsonData)) {
          throw new Error('Invalid data format received from server');
        }

        // Extract unique categories (sorted)
        const uniqueCats = Array.from(
          new Set(jsonData.map((r) => r.incident_category))
        ).sort();
        
        setAllCategories(['All', ...uniqueCats]);
        setCategories(['All', ...uniqueCats]);
      } catch (err) {
        console.error('Error fetching categories:', err);
        setAllCategories(['All']);
        setCategories(['All']);
      }
    };

    fetchAllCategories();
  }, []);

  //
  // 1) Fetch data and extract categories
  //
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Get current date for end date if not specified
        const currentDate = new Date();
        const endDateObj = endDate ? new Date(endDate) : currentDate;
        
        // Format dates for API
        const formattedEndDate = endDateObj.toISOString().split('T')[0];
        
        // Use 'All' as default category if none selected
        const categoryParam = selectedCategory || 'All';
        
        const response = await fetch(
          `${API_BASE_URL}/api/crime-locations?category=${encodeURIComponent(categoryParam)}&end_date=${formattedEndDate}`,
          {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonData = await response.json();
        
        if (!Array.isArray(jsonData)) {
          throw new Error('Invalid data format received from server');
        }

        // filter out invalid coordinates and ensure data is within date range
        const validData = jsonData.filter(item => {
          const lat = parseFloat(item.latitude);
          const lon = parseFloat(item.longitude);
          return !isNaN(lat) && !isNaN(lon) && 
                 lat !== 0 && lon !== 0 &&
                 lat >= -90 && lat <= 90 &&
                 lon >= -180 && lon <= 180;
        });

        if (validData.length === 0) {
          setData([]);
          return;
        }

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
  }, [selectedCategory, endDate]);

  //
  // 2) Initialize the Mapbox map ONCE
  //
  useEffect(() => {
    if (mapRef.current) return;

    mapboxgl.accessToken = 'pk.eyJ1IjoibWVsaW5kYTAzMjYiLCJhIjoiY21iaGNrdDNrMDd4cjJscHFvcWxwa2NsbCJ9.xXrM3t_PX0ezAbpXwI41Cw';

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v11',
      // style: 'mapbox://styles/mapbox/light-v11',
      center: [-122.4194, 37.7749],
      zoom: 12,
      maxZoom: 15
    });

    // Keep a ref to the map instance
    mapRef.current = map;

    map.on('style.load', () => {
      // Add an empty GeoJSON source; we'll populate it when data is loaded
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
            'rgba(0, 0, 255, 0)', 
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
      map.remove();
        mapRef.current = null;
    };
  }, []);

  //
  // 3) Update map source when data changes
  //
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !data.length) return;

    // Wait for the map style to load if needed
    if (!map.isStyleLoaded()) {
      map.once('style.load', updateData);
      return;
    }

    function updateData() {
      const features = data.map(item => ({
      type: 'Feature',
        properties: {},
      geometry: {
        type: 'Point',
          coordinates: [
            parseFloat(item.longitude),
            parseFloat(item.latitude)
          ]
      }
    }));

      const source = map.getSource('incidents');
      if (source) {
        source.setData({
      type: 'FeatureCollection',
      features: features
        });
      }
    }

    updateData();
  }, [data]);

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        {/* ── Header + Dropdown ── */}
        <div style={{ textAlign: 'center', margin: '10px 0' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: '500' }}>
            {selectedCategory === 'All' ? 'Select Category' : ''}
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
        >
          {loading && (
            <div
              style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                padding: '8px 12px',
                borderRadius: '4px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                zIndex: 1000,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '14px',
                color: '#666'
              }}
            >
              <div
                style={{
                  width: '16px',
                  height: '16px',
                  border: '2px solid #666',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}
              />
              Loading map data...
            </div>
          )}
        </div>

        {error && (
          <div style={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%', 
            transform: 'translate(-50%, -50%)',
            background: 'rgba(255, 0, 0, 0.1)',
            padding: '1rem',
            borderRadius: '4px'
          }}>
            {error}
          </div>
        )}
      </div>

      <style>
        {`
          @keyframes spin {
            to {
              transform: rotate(360deg);
            }
          }
        `}
      </style>
    </>
  );
}
