import numpy as np
import face_recognition
import os
import cv2
from datetime import datetime, timedelta
import csv
import time
from threading import Lock, Thread
from queue import Queue
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
from newPerson import addNewPerson

# Existing code...

def atten():
    global running, recognition_thread  # Declare the variable here to use it across functions
    running = False  # Initialize running to False at the start
    recognition_thread = None  # Initialize the thread variable

    # Path to the dataset
    path = 'Dataset'
    images = []
    classNames = []

    # List all image files in the dataset folder
    myList = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # Only extract the name without the file extension

    def findEncodings(images):
        encodeList = []
        for img in images:
            if img is None:
                print("One of the images is invalid. Skipping...")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                encode = encodings[0]
                encodeList.append(encode)
            else:
                print("No face found in one of the images. Skipping.")
        return encodeList

    # Create attendance CSV if it doesn't exist
    file_path = 'Attendance.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time'])

    file_lock = Lock()
    last_attendance_times = {}
    ATTENDANCE_INTERVAL = 10  # seconds

    def markAttendance(name):
        current_time = datetime.now()
        with file_lock:
            with open(file_path, 'r+') as f:
                myDataList = f.readlines()
                nameList = [entry.split(',')[0] for entry in myDataList]
                if (name not in last_attendance_times or
                        current_time - last_attendance_times[name] >= timedelta(seconds=ATTENDANCE_INTERVAL)):
                    dtString = current_time.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dtString}')
                    last_attendance_times[name] = current_time
                    print(f"Attendance marked for {name} at {dtString}")
                else:
                    time_remaining = ATTENDANCE_INTERVAL - (current_time - last_attendance_times[name]).total_seconds()
                    print(f"{name} must wait {int(time_remaining)} seconds for the next attendance.")

    encodeListKnown = findEncodings(images)
    print('Encoding Done')

    MIN_CONFIDENCE = 0.6
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture")
        return

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Attendance System")

    # Create a label to display the video feed
    label = tk.Label(root)
    label.pack()

    frame_queue = Queue()  # Create a queue to hold frames for processing

    def recognize_faces():
        global running
        while running:
            success, img = video_capture.read()
            if not success or img is None:
                print("Failed to capture image. Retrying...")
                continue

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS)

            for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDist)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    confidence = 1 - faceDist[matchIndex]

                    if confidence >= MIN_CONFIDENCE:
                        name_accu = name + ',' + str(int(confidence * 100)) + '%'
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2 + 60, y2), (0, 0, 0), cv2.FILLED)
                        cv2.putText(img, name_accu, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        markAttendance(name)

            # Put the processed frame in the queue for the main thread to use
            frame_queue.put(img)

    def start_recognition():
        global recognition_thread, running  # Declare global variables here
        if not running:  # Start the thread only if it's not already running
            running = True
            recognition_thread = Thread(target=recognize_faces)
            recognition_thread.start()
            print("Face recognition started.")

    def stop_recognition():
        global running  # Declare global variable here
        running = False
        if recognition_thread is not None:
            recognition_thread.join()  # Wait for the thread to finish
            print("Face recognition stopped.")

    def add_new_person():
        # Prompt for the name
        name = simpledialog.askstring("Input", "Enter the person's name:")
        if name:
            # Capture an image using the webcam
            success, image = video_capture.read()
            if success:
                addNewPerson(image, name)  # Call the addNewPerson function with the captured image and name
            else:
                print("Failed to capture image. Please try again.")

    def show_frame():
        if not frame_queue.empty():  # Check if there are frames in the queue
            img = frame_queue.get()  # Get the latest frame
            # Convert the image to a Tkinter-compatible format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the label with the new frame
            label.img_tk = img_tk
            label.config(image=img_tk)

        root.after(10, show_frame)  # Schedule the next frame update

    # Start the video feed and recognition
    start_recognition()
    show_frame()

    # Add buttons for controlling recognition
    start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
    start_button.pack()

    stop_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
    stop_button.pack()

    add_person_button = tk.Button(root, text="Add New Person", command=add_new_person)
    add_person_button.pack()

    # Start the Tkinter main loop
    root.mainloop()

    # Release resources when the window is closed
    stop_recognition()  # Stop the recognition thread
    video_capture.release()
    print("Video capture and resources released properly.")

if __name__ == '__main__':
    # Call the atten function to start the application
    atten()
