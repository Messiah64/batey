import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HeadsetAttentionDisplay = () => {
  const [ranges, setRanges] = useState({
    high: 75,
    medium: 50,
    low: 25
  });
  const [liveAttention, setLiveAttention] = useState(0);
  const [position, setPosition] = useState({ x: 0, y: 0, rotation: 0});

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

  useEffect(() => {
    const newPosition = getPosition(liveAttention);
    setPosition(newPosition);
  }, [liveAttention, ranges]);

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

  const getPosition = (value) => {
    const containerSize = 64;
    const maxMove = containerSize / 4;

    if (value < ranges.low) {
      // Blue range (bottom)
      return { x: 0, y: maxMove, rotation: 0};
    } else if (value < ranges.medium) {
      // Green range (left)
      return { x: -maxMove, y: 0, rotation: -90};
    } else if (value < ranges.high) {
      // Yellow range (right)
      return { x: maxMove, y: 0, rotation: 90};
    } else {
      // Red range (top)
      return { x: 0, y: -maxMove, rotation: 0};
    }
  };

  const getActiveWave = (value) => {
    if (value >= ranges.high) return 'red';
    if (value >= ranges.medium) return 'yellow';
    if (value >= ranges.low) return 'green';
    return 'blue';
  };

  const getWaveIntensity = (value) => {
    const activeWave = getActiveWave(value);
    let intensity;
    switch (activeWave) {
      case 'red':
        intensity = (value - ranges.high) / (100 - ranges.high);
        break;
      case 'yellow':
        intensity = (value - ranges.medium) / (ranges.high - ranges.medium);
        break;
      case 'green':
        intensity = (value - ranges.low) / (ranges.medium - ranges.low);
        break;
      case 'blue':
        intensity = value / ranges.low;
        break;
      default:
        intensity = 0;
    }
    return Math.min(Math.max(intensity, 0), 1);
  };

  const getImageData = (value) => {
    const activeWave = getActiveWave(value);
    const intensity = getWaveIntensity(value);
    
    const baseImages = {
      blue: './vector (2).png',
      green: './vector.png',
      yellow: './vector (3).png',
      red: './Vector (1).png'
    };

    const transitionImages = {
      blue: './vector.png',
      green: './vector (3).png',
      yellow: './Vector (1).png',
      red: './Vector (1).png'  // No transition for the highest level
    };

    const baseImage = baseImages[activeWave];
    const transitionImage = transitionImages[activeWave];

    return {
      baseImage,
      transitionImage,
      opacity: intensity
    };
  };

  const activeWave = getActiveWave(liveAttention);
  const waveIntensity = getWaveIntensity(liveAttention);
  const imageData = getImageData(liveAttention);

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-2xl font-bold text-center">Attention Display</h2>
      
      <div className="text-center">
        <h3 className="text-lg font-semibold">Live Attention Value: {liveAttention}</h3>
      </div>

      <div className="relative w-64 h-64 mx-auto">
        {/* Headset container */}
        <div 
            className="absolute inset-0 flex items-center justify-center z-10 transition-all duration-300 ease-out"
            style={{
                transform: `translate(${position.x}px, ${position.y}px) rotate(${position.rotation}deg)`
            }}
            >
          <img 
            src={imageData.baseImage}
            alt="Headset Base" 
            className="w-40 h-40 object-contain absolute"
          />
          <img 
            src={imageData.transitionImage}
            alt="Headset Transition" 
            className="w-40 h-40 object-contain absolute transition-opacity duration-300"
            style={{ opacity: imageData.opacity }}
          />
        </div>
        
        {/* Wave container */}
        <div className="absolute inset-0">
          <div className={`wave ${activeWave}-wave`} style={{
            animationDuration: `${2 / waveIntensity}s`,
            opacity: waveIntensity * 0.7
          }}></div>
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
            <div key={i} className={`w-1 h-full ${getActiveWave(i + 1)}-bg`}></div>
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
        <p>Full Forward (Red - Top): value >= {ranges.high}</p>
        <p>Right (Yellow - Right): {ranges.high} > value >= {ranges.medium}</p>
        <p>Left (Green - Left): {ranges.medium} > value >= {ranges.low}</p>
        <p>Stop (Blue - Bottom): {ranges.low} > value >= 0</p>
      </div>

      <style jsx>{`
        @keyframes singlePulse {
          0% { transform: scale(0); opacity: 0.7; }
          100% { transform: scale(1); opacity: 0; }
        }

        .wave {
          position: absolute;
          animation: singlePulse 2s ease-out infinite;
        }

        .red-wave {
          top: 0px;
          left: 0%;
          right: 0%;
          height: 60px;
          border-radius: 50% 50% 0 0;
          background-color: rgba(255, 0, 0, 0.5);
        }

        .yellow-wave {
          right: 0px;
          top: 0%;
          bottom: 0%;
          width: 60px;
          border-radius: 0 50% 50% 0;
          background-color: rgba(255, 255, 0, 0.5);
        }

        .green-wave {
          left: 0px;
          top: 0%;
          bottom: 0%;
          width: 60px;
          border-radius: 50% 0 0 50%;
          background-color: rgba(0, 255, 0, 0.5);
        }

        .blue-wave {
          bottom: 0px;
          left: 0%;
          right: 0%;
          height: 60px;
          border-radius: 0 0 50% 50%;
          background-color: rgba(0, 0, 255, 0.5);
        }

        .red-bg { background-color: red; }
        .yellow-bg { background-color: yellow; }
        .green-bg { background-color: green; }
        .blue-bg { background-color: blue; }
      `}</style>
    </div>
  );
};

export default HeadsetAttentionDisplay;