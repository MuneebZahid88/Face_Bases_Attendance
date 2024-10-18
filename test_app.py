import cv2
import os
from att import atten  # Assuming this function uses Tkinter
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_cors import CORS
from newPerson import addNewPerson
from threading import Thread

# Flask setup
app = Flask(__name__)
CORS(app)

captured_image_path = 'captured_image.png'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/newPerson', methods=['POST'])
def add_new_person():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    ret, img = video_capture.read()
    if ret:
        cv2.imwrite(captured_image_path, img)
        video_capture.release()
        try:
            addNewPerson(img, name)
            return jsonify({"message": f"New person '{name}' has been successfully registered."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        video_capture.release()
        return jsonify({"error": "Failed to capture image"}), 500

@app.route('/capture_image', methods=['GET'])
def capture_image():
    return send_file(captured_image_path, mimetype='image/jpeg')

@app.route('/attendance')
def attendance():
    atten()  # Calling the Tkinter-based function
    return redirect(url_for('home'))

# Function to run the Flask server
def run_flask():
    app.run(debug=True, use_reloader=False)  # Disable reloader to prevent Flask from starting twice

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Now run Tkinter on the main thread
    atten()  # Assuming this is where Tkinter is running
