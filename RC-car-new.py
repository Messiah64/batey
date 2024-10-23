import time
import asyncio
import numpy as np
import websockets
from sklearn.svm import OneClassSVM
from neuropy3.neuropy3 import MindWave
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
from collections import deque
from scipy.signal import medfilt

app = Flask(__name__)
CORS(app)

# WebSocket URI for ESP32 (replace with your ESP32's IP address)
ESP32_URI = "ws://172.20.10.2:81"

# Define EEG features
FEATURES = ['delta', 'theta', 'alpha_l', 'alpha_h', 'beta_l', 'beta_h']

# Global variables
model = None
last_function_call_time = time.time()
ranges = {
    'high': 75,
    'medium': 50,
    'low': 25
}
latest_attention_value = 0
signal_processor = None

class AttentionSignalProcessor:
    def __init__(self, 
                 window_size=10,
                 median_window=5,
                 threshold_change=20,
                 min_stable_duration=1.0):
        """
        Initialize the signal processor with filtering parameters.
        
        Args:
            window_size: Size of moving average window
            median_window: Size of median filter window (must be odd)
            threshold_change: Maximum allowed change between consecutive filtered values
            min_stable_duration: Minimum time (seconds) to maintain a state
        """
        self.window_size = window_size
        self.median_window = median_window
        self.threshold_change = threshold_change
        self.min_stable_duration = min_stable_duration
        
        self.attention_buffer = deque(maxlen=window_size)
        self.last_filtered_value = None
        self.last_state_change = 0
        self.current_state = None
    
    def moving_average_filter(self, value):
        """Apply moving average filter to smooth the signal."""
        self.attention_buffer.append(value)
        return np.mean(self.attention_buffer)
    
    def median_filter(self, values):
        """Apply median filter to remove spikes."""
        return float(medfilt(np.array(values), self.median_window)[0])
    
    def hysteresis_filter(self, value, ranges):
        """Apply hysteresis to prevent rapid state changes."""
        current_time = time.time()
        
        # Determine the new state based on the new logic
        if value > ranges['high']:
            new_state = 'B'
        elif value > ranges['medium']:
            new_state = 'C'
        elif value > ranges['low']:
            new_state = 'A'
        else:
            new_state = 'D'
        
        # Check if enough time has passed since last state change
        if (self.current_state != new_state and 
            current_time - self.last_state_change >= self.min_stable_duration):
            self.current_state = new_state
            self.last_state_change = current_time
            return new_state
        
        return self.current_state
    
    def process_attention(self, value, ranges):
        """Process the attention value through all filters."""
        # Apply moving average
        ma_value = self.moving_average_filter(value)
        
        # Apply median filter if we have enough values
        if len(self.attention_buffer) >= self.median_window:
            med_value = self.median_filter(list(self.attention_buffer))
        else:
            med_value = ma_value
        
        # Limit rate of change
        if self.last_filtered_value is not None:
            change = med_value - self.last_filtered_value
            if abs(change) > self.threshold_change:
                med_value = self.last_filtered_value + np.sign(change) * self.threshold_change
        
        self.last_filtered_value = med_value
        
        # Apply hysteresis for state determination
        state = self.hysteresis_filter(med_value, ranges)
        
        return med_value, state

def extract_features(eeg_data):
    """Extract features from EEG data."""
    return np.array([eeg_data[f] for f in FEATURES]).reshape(1, -1)

async def send_command_via_websocket(command):
    """Send command to ESP32 via WebSocket."""
    try:
        async with websockets.connect(ESP32_URI) as websocket:
            await websocket.send(command)
            print(f"Sent command via WebSocket: {command}")
    except Exception as e:
        print(f"Failed to send command via WebSocket: {e}")

def A():
    print(f"Function A: Attention is high (more than {ranges['high']})")
    asyncio.run(send_command_via_websocket('A'))

def B():
    print(f"Function B: Attention is medium-high (more than {ranges['medium']} but less than or equal to {ranges['high']})")
    asyncio.run(send_command_via_websocket('B'))

def C():
    print(f"Function C: Attention is medium-low (more than {ranges['low']} but less than or equal to {ranges['medium']})")
    asyncio.run(send_command_via_websocket('C'))

def D():
    print(f"Function D: Attention is low (more than 0 but less than or equal to {ranges['low']})")
    asyncio.run(send_command_via_websocket('D'))

def eeg_callback(data):
    """Callback function to handle EEG data."""
    global model

    features = extract_features(data)

    if model is None:
        initial_data = []
        initial_data.append(features.flatten())

        if len(initial_data) >= 10:
            model = OneClassSVM(nu=0.1)
            model.fit(initial_data)
            print("Model trained on initial data")
    else:
        prediction = model.predict(features)
        if prediction[0] == 1:
            print(f"EEG: {data}")
        else:
            print("EEG data classified as erratic, ignoring.")

def meditation_callback(value):
    """Callback function to handle meditation data."""
    print("Meditation: ", value)

def attention_callback(value):
    """Callback function to handle attention data with signal processing."""
    global last_function_call_time, ranges, latest_attention_value, signal_processor
    
    # Process the attention value
    filtered_value, state = signal_processor.process_attention(value, ranges)
    latest_attention_value = filtered_value
    
    print(f"Raw Attention: {value}, Filtered: {filtered_value:.1f}, State: {state}")
    
    # Execute state function if enough time has passed
    current_time = time.time()
    if current_time - last_function_call_time >= 1:
        if state == 'B':
            B()
        elif state == 'C':
            C()
        elif state == 'A':
            A()
        elif state == 'D':
            D()
        last_function_call_time = current_time

@app.route('/get_attention', methods=['GET'])
def get_attention():
    """Flask route to get current attention value."""
    return jsonify({"attention": latest_attention_value}), 200

@app.route('/update_ranges', methods=['POST'])
def update_ranges():
    """Flask route to update attention ranges."""
    global ranges
    new_ranges = request.json
    ranges['high'] = new_ranges.get('high', ranges['high'])
    ranges['medium'] = new_ranges.get('medium', ranges['medium'])
    ranges['low'] = new_ranges.get('low', ranges['low'])
    print(f"Ranges updated: {ranges}")
    return jsonify({"message": "Ranges updated successfully"}), 200

@app.route('/get_ranges', methods=['GET'])
def get_ranges():
    """Flask route to get current ranges."""
    return jsonify(ranges), 200

def run_flask():
    """Run Flask server."""
    app.run(debug=False, port=5000)

def main():
    """Main function to run the application."""
    global last_function_call_time, signal_processor
    
    # Initialize signal processor with default parameters
    signal_processor = AttentionSignalProcessor(
        window_size=10,      # Adjust this value for more/less smoothing
        median_window=5,     # Must be odd number, higher removes more spikes
        threshold_change=20,  # Maximum allowed change between readings
        min_stable_duration=1.0  # Minimum time to stay in a state
    )
    
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True  # This ensures the thread will be closed when the main program exits
    flask_thread.start()

    # Initialize the MindWave device
    mw = MindWave(address='A4:DA:32:70:03:4E', autostart=False, verbose=3)

    # Set up callbacks
    mw.set_callback('eeg', eeg_callback)
    mw.set_callback('meditation', meditation_callback)
    mw.set_callback('attention', attention_callback)

    # Start the device
    mw.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    finally:
        print("Cleaning up...")
        mw.stop()
        # The Flask thread will automatically close due to daemon=True

if __name__ == "__main__":
    main()
