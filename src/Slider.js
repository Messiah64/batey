import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AttentionRangeSlider = () => {
  const [ranges, setRanges] = useState({
    high: 75,
    medium: 50,
    low: 25
  });
  const [liveAttention, setLiveAttention] = useState(0); // New state for live attention

  useEffect(() => {
    // Fetch initial ranges from the server
    axios.get('http://localhost:5000/get_ranges')
      .then(response => {
        setRanges(response.data);
      })
      .catch(error => {
        console.error('Error fetching ranges:', error);
      });

    // Fetch live attention value every second
    const interval = setInterval(() => {
      axios.get('http://localhost:5000/get_attention')
        .then(response => {
          setLiveAttention(response.data.attention);
        })
        .catch(error => {
          console.error('Error fetching live attention:', error);
        });
    }, 1000);

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  const handleSliderChange = (event) => {
    const { name, value } = event.target;
    const newRanges = {
      ...ranges,
      [name]: parseInt(value)
    };
    setRanges(newRanges);

    // Send updated ranges to the server
    axios.post('http://localhost:5000/update_ranges', newRanges)
      .then(response => {
        console.log('Ranges updated successfully');
      })
      .catch(error => {
        console.error('Error updating ranges:', error);
      });
  };

  const getColor = (value) => {
    if (value >= ranges.high) return 'bg-red-500';
    if (value >= ranges.medium) return 'bg-yellow-500';
    if (value >= ranges.low) return 'bg-green-500';
    return 'bg-blue-500';
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-2xl font-bold text-center">Attention Range Adjuster</h2>
      
      {/* Display live attention value */}
      <div className="text-center">
        <h3 className="text-lg font-semibold">Live Attention Value: {liveAttention}</h3>
      </div>

      {/* New section for live attention visualization */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Live Attention Visualization</h3>
        <div className="w-full h-8 bg-gray-200 rounded-lg">
          <div
            style={{ width: `${liveAttention}%` }}
            className={`h-full ${getColor(liveAttention)}`}
          ></div>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span>0</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
      </div>

      {Object.entries(ranges).map(([key, value]) => (
        <div key={key} className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            {key.charAt(0).toUpperCase() + key.slice(1)} Threshold: {value}
          </label>
          <input
            type="range"
            name={key}
            min="0"
            max="100"
            value={value}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      ))}

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Range Visualization</h3>
        <div className="w-full h-8 flex">
          {[...Array(100)].map((_, i) => (
            <div key={i} className={`w-1 h-full ${getColor(i + 1)}`}></div>
          ))}
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span>0</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
      </div>

      <div className="mt-6 space-y-2">
        <h3 className="text-lg font-semibold">Current Ranges:</h3>
        <p>A(): value > {ranges.high}</p>
        <p>B(): {ranges.high} >= value > {ranges.medium}</p>
        <p>C(): {ranges.medium} >= value > {ranges.low}</p>
        <p>D(): {ranges.low} >= value >= 0</p>
      </div>
    </div>
  );
};

export default AttentionRangeSlider;
