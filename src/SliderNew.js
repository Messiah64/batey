import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HeadsetAttentionDisplay = () => {
  const [ranges, setRanges] = useState({
    high: 75,
    medium: 50,
    low: 25
  });
  const [liveAttention, setLiveAttention] = useState(0);

  useEffect(() => {
    axios.get('http://localhost:5000/get_ranges')
      .then(response => {
        setRanges(response.data);
      })
      .catch(error => {
        console.error('Error fetching ranges:', error);
      });

    const interval = setInterval(() => {
      axios.get('http://localhost:5000/get_attention')
        .then(response => {
          setLiveAttention(response.data.attention);
        })
        .catch(error => {
          console.error('Error fetching live attention:', error);
        });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleSliderChange = (event) => {
    const { name, value } = event.target;
    const newRanges = {
      ...ranges,
      [name]: parseInt(value)
    };
    setRanges(newRanges);

    axios.post('http://localhost:5000/update_ranges', newRanges)
      .then(response => {
        console.log('Ranges updated successfully');
      })
      .catch(error => {
        console.error('Error updating ranges:', error);
      });
  };

  const getColor = (value) => {
    if (value >= ranges.high) return 'red';
    if (value >= ranges.medium) return 'yellow';
    if (value >= ranges.low) return 'green';
    return 'blue';
  };

  const getWavePosition = (color) => {
    switch (color) {
      case 'red': return 'top';
      case 'yellow': return 'right';
      case 'green': return 'left';
      default: return 'none';
    }
  };

  const color = getColor(liveAttention);
  const wavePosition = getWavePosition(color);

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-2xl font-bold text-center">Attention Display</h2>
      
      <div className="text-center">
        <h3 className="text-lg font-semibold">Live Attention Value: {liveAttention}</h3>
      </div>

      <div className="relative w-64 h-64 mx-auto">
        <img 
          src="/path/to/your/headset-image.png" 
          alt="Headset" 
          className="w-full h-full object-contain"
        />
        
        {wavePosition !== 'none' && (
          <div className={`absolute inset-0 overflow-hidden ${wavePosition}-wave`}>
            <div className={`w-full h-full ${color}-wave`}></div>
          </div>
        )}
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
            <div key={i} className={`w-1 h-full ${getColor(i + 1)}-bg`}></div>
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
        <p>Full Forward (Red): value > {ranges.high}</p>
        <p>Right (Yellow): {ranges.high} >= value > {ranges.medium}</p>
        <p>Left (Green): {ranges.medium} >= value > {ranges.low}</p>
        <p>Stop (Blue): {ranges.low} >= value >= 0</p>
      </div>

      <style jsx>{`
        @keyframes pulsate {
          0% { transform: scale(0); opacity: 1; }
          100% { transform: scale(3); opacity: 0; }
        }

        .top-wave > div {
          border-radius: 50% 50% 0 0;
          height: 50%;
          top: 0;
          left: 25%;
          right: 25%;
          position: absolute;
          animation: pulsate 2s ease-out infinite;
        }

        .left-wave > div {
          border-radius: 50% 0 0 50%;
          width: 50%;
          top: 25%;
          bottom: 25%;
          left: 0;
          position: absolute;
          animation: pulsate 2s ease-out infinite;
        }

        .right-wave > div {
          border-radius: 0 50% 50% 0;
          width: 50%;
          top: 25%;
          bottom: 25%;
          right: 0;
          position: absolute;
          animation: pulsate 2s ease-out infinite;
        }

        .red-wave { background-color: rgba(255, 0, 0, 0.5); }
        .yellow-wave { background-color: rgba(255, 255, 0, 0.5); }
        .green-wave { background-color: rgba(0, 255, 0, 0.5); }
        .blue-wave { background-color: rgba(0, 0, 255, 0.5); }

        .red-bg { background-color: red }
        .yellow-bg { background-color: yellow; }
        .green-bg { background-color: green; }
        .blue-bg { background-color: blue; }
      `}</style>
    </div>
  );
};

export default HeadsetAttentionDisplay;