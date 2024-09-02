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
import subprocess
import psutil

app = Flask(__name__)
CORS(app)

eeg_data = [
    {'delta': 629242, 'theta': 461641, 'alpha_l': 72414, 'alpha_h': 20509, 'beta_l': 121180, 'beta_h': 69509, 'gamma_l': 18474, 'gamma_m': 2194},
    {'delta': 1141372, 'theta': 45655, 'alpha_l': 17259, 'alpha_h': 10639, 'beta_l': 11846, 'beta_h': 6828, 'gamma_l': 1087, 'gamma_m': 848},
    {'delta': 66755, 'theta': 20911, 'alpha_l': 797, 'alpha_h': 477, 'beta_l': 998, 'beta_h': 1440, 'gamma_l': 470, 'gamma_m': 414},
    {'delta': 1146726, 'theta': 186393, 'alpha_l': 60627, 'alpha_h': 13696, 'beta_l': 25124, 'beta_h': 19363, 'gamma_l': 3487, 'gamma_m': 873},
    {'delta': 731162, 'theta': 15139, 'alpha_l': 3939, 'alpha_h': 2967, 'beta_l': 491, 'beta_h': 1205, 'gamma_l': 181, 'gamma_m': 88},
    {'delta': 1239331, 'theta': 20050, 'alpha_l': 3261, 'alpha_h': 1206, 'beta_l': 641, 'beta_h': 1434, 'gamma_l': 650, 'gamma_m': 1215},
    {'delta': 115458, 'theta': 142765, 'alpha_l': 18911, 'alpha_h': 23483, 'beta_l': 24583, 'beta_h': 41970, 'gamma_l': 5257, 'gamma_m': 2133},
    {'delta': 220010, 'theta': 20133, 'alpha_l': 20058, 'alpha_h': 2971, 'beta_l': 4399, 'beta_h': 10094, 'gamma_l': 5105, 'gamma_m': 2338},
    {'delta': 85498, 'theta': 15083, 'alpha_l': 6187, 'alpha_h': 4939, 'beta_l': 1885, 'beta_h': 3266, 'gamma_l': 6727, 'gamma_m': 1585},
    {'delta': 35880, 'theta': 32479, 'alpha_l': 1583, 'alpha_h': 7280, 'beta_l': 7358, 'beta_h': 6541, 'gamma_l': 3541, 'gamma_m': 959},
    {'delta': 825579, 'theta': 73217, 'alpha_l': 5501, 'alpha_h': 9547, 'beta_l': 6885, 'beta_h': 4704, 'gamma_l': 1064, 'gamma_m': 125},
    {'delta': 67706, 'theta': 3347, 'alpha_l': 1348, 'alpha_h': 530, 'beta_l': 1512, 'beta_h': 620, 'gamma_l': 494, 'gamma_m': 242},
    {'delta': 603965, 'theta': 15544, 'alpha_l': 1533, 'alpha_h': 1993, 'beta_l': 1712, 'beta_h': 3211, 'gamma_l': 1099, 'gamma_m': 484},
    {'delta': 569152, 'theta': 17234, 'alpha_l': 7552, 'alpha_h': 5798, 'beta_l': 3263, 'beta_h': 2010, 'gamma_l': 806, 'gamma_m': 473},
    {'delta': 159903, 'theta': 6609, 'alpha_l': 1669, 'alpha_h': 1248, 'beta_l': 431, 'beta_h': 957, 'gamma_l': 307, 'gamma_m': 326},
    {'delta': 148347, 'theta': 12322, 'alpha_l': 2260, 'alpha_h': 13359, 'beta_l': 2492, 'beta_h': 3323, 'gamma_l': 1189, 'gamma_m': 1113},
    {'delta': 625619, 'theta': 33752, 'alpha_l': 12883, 'alpha_h': 14163, 'beta_l': 9192, 'beta_h': 7586, 'gamma_l': 4350, 'gamma_m': 774},
    {'delta': 862662, 'theta': 133119, 'alpha_l': 14025, 'alpha_h': 29846, 'beta_l': 21867, 'beta_h': 37299, 'gamma_l': 6566, 'gamma_m': 7289},
    {'delta': 377036, 'theta': 6880, 'alpha_l': 2108, 'alpha_h': 1454, 'beta_l': 1259, 'beta_h': 325, 'gamma_l': 164, 'gamma_m': 82},
    {'delta': 289974, 'theta': 7213, 'alpha_l': 2912, 'alpha_h': 649, 'beta_l': 798, 'beta_h': 1592, 'gamma_l': 1186, 'gamma_m': 598},
    {'delta': 438963, 'theta': 7552, 'alpha_l': 13420, 'alpha_h': 18915, 'beta_l': 5614, 'beta_h': 2579, 'gamma_l': 1146, 'gamma_m': 770},
    {'delta': 101332, 'theta': 43212, 'alpha_l': 10404, 'alpha_h': 5805, 'beta_l': 9393, 'beta_h': 9450, 'gamma_l': 3568, 'gamma_m': 1394},
    {'delta': 54357, 'theta': 22632, 'alpha_l': 6976, 'alpha_h': 1415, 'beta_l': 2305, 'beta_h': 7981, 'gamma_l': 9985, 'gamma_m': 2054},
    {'delta': 88077, 'theta': 132843, 'alpha_l': 43403, 'alpha_h': 6795, 'beta_l': 13101, 'beta_h': 5687, 'gamma_l': 7907, 'gamma_m': 6592},
    {'delta': 161911, 'theta': 13774, 'alpha_l': 12782, 'alpha_h': 7773, 'beta_l': 6111, 'beta_h': 7809, 'gamma_l': 13451, 'gamma_m': 2009},
    {'delta': 761005, 'theta': 84018, 'alpha_l': 12061, 'alpha_h': 8028, 'beta_l': 21790, 'beta_h': 12457, 'gamma_l': 906, 'gamma_m': 1933},
    {'delta': 85498, 'theta': 15083, 'alpha_l': 6187, 'alpha_h': 4939, 'beta_l': 1885, 'beta_h': 3266, 'gamma_l': 6727, 'gamma_m': 1585},
    {'delta': 35880, 'theta': 32479, 'alpha_l': 1583, 'alpha_h': 7280, 'beta_l': 7358, 'beta_h': 6541, 'gamma_l': 3541, 'gamma_m': 959},
    {'delta': 825579, 'theta': 73217, 'alpha_l': 5501, 'alpha_h': 9547, 'beta_l': 6885, 'beta_h': 4704, 'gamma_l': 1064, 'gamma_m': 125},
    {'delta': 67706, 'theta': 3347, 'alpha_l': 1348, 'alpha_h': 530, 'beta_l': 1512, 'beta_h': 620, 'gamma_l': 494, 'gamma_m': 242},
    {'delta': 603965, 'theta': 15544, 'alpha_l': 1533, 'alpha_h': 1993, 'beta_l': 1712, 'beta_h': 3211, 'gamma_l': 1099, 'gamma_m': 484},
    {'delta': 569152, 'theta': 17234, 'alpha_l': 7552, 'alpha_h': 5798, 'beta_l': 3263, 'beta_h': 2010, 'gamma_l': 806, 'gamma_m': 473},
    {'delta': 159903, 'theta': 6609, 'alpha_l': 1669, 'alpha_h': 1248, 'beta_l': 431, 'beta_h': 957, 'gamma_l': 307, 'gamma_m': 326},
    {'delta': 148347, 'theta': 12322, 'alpha_l': 2260, 'alpha_h': 13359, 'beta_l': 2492, 'beta_h': 3323, 'gamma_l': 1189, 'gamma_m': 1113},
    {'delta': 625619, 'theta': 33752, 'alpha_l': 12883, 'alpha_h': 14163, 'beta_l': 9192, 'beta_h': 7586, 'gamma_l': 4350, 'gamma_m': 774},
    {'delta': 862662, 'theta': 133119, 'alpha_l': 14025, 'alpha_h': 29846, 'beta_l': 21867, 'beta_h': 37299, 'gamma_l': 6566, 'gamma_m': 7289},
    {'delta': 377036, 'theta': 6880, 'alpha_l': 2108, 'alpha_h': 1454, 'beta_l': 1259, 'beta_h': 325, 'gamma_l': 164, 'gamma_m': 82},
    {'delta': 289974, 'theta': 7213, 'alpha_l': 2912, 'alpha_h': 649, 'beta_l': 798, 'beta_h': 1592, 'gamma_l': 1186, 'gamma_m': 598},
    {'delta': 438963, 'theta': 7552, 'alpha_l': 13420, 'alpha_h': 18915, 'beta_l': 5614, 'beta_h': 2579, 'gamma_l': 1146, 'gamma_m': 770},
    {'delta': 101332, 'theta': 43212, 'alpha_l': 10404, 'alpha_h': 5805, 'beta_l': 9393, 'beta_h': 9450, 'gamma_l': 3568, 'gamma_m': 1394},
    {'delta': 54357, 'theta': 22632, 'alpha_l': 6976, 'alpha_h': 1415, 'beta_l': 2305, 'beta_h': 7981, 'gamma_l': 9985, 'gamma_m': 2054},
    {'delta': 88077, 'theta': 132843, 'alpha_l': 43403, 'alpha_h': 6795, 'beta_l': 13101, 'beta_h': 5687, 'gamma_l': 7907, 'gamma_m': 6592},
    {'delta': 161911, 'theta': 13774, 'alpha_l': 12782, 'alpha_h': 7773, 'beta_l': 6111, 'beta_h': 7809, 'gamma_l': 13451, 'gamma_m': 2009},
    {'delta': 761005, 'theta': 84018, 'alpha_l': 12061, 'alpha_h': 8028, 'beta_l': 21790, 'beta_h': 12457, 'gamma_l': 906, 'gamma_m': 1933},
    {'delta': 85498, 'theta': 15083, 'alpha_l': 6187, 'alpha_h': 4939, 'beta_l': 1885, 'beta_h': 3266, 'gamma_l': 6727, 'gamma_m': 1585}
]
# WebSocket URI for ESP32 (replace with your ESP32's IP address)
ESP32_URI = "ws://192.168.10.147:81"  # Replace with the actual IP address

# Define EEG features (can be adjusted based on your data)
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

# Function to extract features from EEG data
def extract_features(eeg_data):
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

# Function to open Google Chrome on Windows
def open_google_chrome():
    # Function to check if Google Chrome is already running
    def is_chrome_running():
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'].lower() == 'chrome.exe':
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    try:
        if not is_chrome_running():
            # Use the 'start' command to open Google Chrome with a specific link
            url = "https://splms.polite.edu.sg/d2l/home"  # Replace with your desired URL
            subprocess.run(["start", "chrome", url], shell=True)
            print(f"Google Chrome opened successfully with URL: {url}")
        else:
            print("Google Chrome is already running.")
    except Exception as e:
        print(f"Failed to open Google Chrome: {e}")

def D():
    print(f"Function D: Attention is low (more than 0 but less than or equal to {ranges['low']})")
    asyncio.run(send_command_via_websocket('D'))
    open_google_chrome()

# Callback function to handle EEG data
def eeg_callback(data):
    global model

    # Extract features from EEG data
    features = extract_features(data)

    # If model is not trained, train it on initial data (you should collect enough data points first)
    if model is None:
        # Initialize an empty list for initial data
        initial_data = []

        # Collect initial data for training
        initial_data.append(features.flatten())

        # Train the model once we have enough data points
        if len(initial_data) >= 10:  # Example threshold; adjust as needed
            model = OneClassSVM(nu=0.1)
            model.fit(initial_data)
            print("Model trained on initial data")
    else:
        # Predict if the model is trained
        prediction = model.predict(features)

        # Only process data classified as normal
        if prediction[0] == 1:
            print(f"EEG: {data}")
        else:
            print("EEG data classified as erratic, ignoring.")

# Callback function to handle meditation data
def meditation_callback(value):
    print("Meditation: ", value)

# Callback function to handle attention data
def attention_callback(value):
    global last_function_call_time, ranges

    print("Attention: ", value)

    current_time = time.time()
    if current_time - last_function_call_time >= 1:
        if value > ranges['high']:
            A()
        elif value > ranges['medium']:
            B()
        elif value > ranges['low']:
            C()
        else:
            D()
        last_function_call_time = current_time

@app.route('/get_attention', methods=['GET'])
def get_attention():
    return jsonify({"attention": latest_attention_value}), 200

# Flask routes
@app.route('/update_ranges', methods=['POST'])
def update_ranges():
    global ranges
    new_ranges = request.json
    ranges['high'] = new_ranges.get('high', ranges['high'])
    ranges['medium'] = new_ranges.get('medium', ranges['medium'])
    ranges['low'] = new_ranges.get('low', ranges['low'])
    print(f"Ranges updated: {ranges}")
    return jsonify({"message": "Ranges updated successfully"}), 200

@app.route('/get_ranges', methods=['GET'])
def get_ranges():
    return jsonify(ranges), 200

def run_flask():
    app.run(debug=False, port=5000)

def main():
    global last_function_call_time

    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Initialize the MindWave device
    mw = MindWave(address='A4:DA:32:70:03:4E', autostart=False, verbose=3)

    # Set up callbacks
    mw.set_callback('eeg', eeg_callback)
    mw.set_callback('meditation', meditation_callback)
    mw.set_callback('attention', attention_callback)

    # Start the device and collect data
    mw.start()

    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script interrupted and stopped.")
    finally:
        mw.stop()
        # Optionally, you can add a way to stop the Flask thread here

if __name__ == "__main__":
    main()