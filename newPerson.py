import numpy as np
import face_recognition
import os
import cv2
import csv
from datetime import datetime



# Path to the dataset
dataset_path = 'Dataset'
csv_file_path = 'people_data.csv'




# Function to create CSV file if it does not exist
def create_csv_file():
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Name', 'Date', 'Time', 'Picture']
            writer.writerow(headers)

# Function to add a new person and save the image
def addNewPerson(image, name):
    # Resize the image to a suitable size for face detection
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image to RGB
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    # Attempt to find face encodings
    new_encodings = face_recognition.face_encodings(rgb_small_image)

    if len(new_encodings) > 0:
        new_encoding = new_encodings[0]

        # Save the image in the dataset
        image_path = os.path.join(dataset_path, f'{name}.png')
        cv2.imwrite(image_path, image)

        print(f"New person {name} added to the dataset.")
        
        # Save the information to the CSV file
        save_to_csv(name, image_path)
    else:
        print("No face detected. Please try again with a clearer image.")

# Function to save information to the CSV file
def save_to_csv(name, pic_path):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [name, date, time, pic_path]
        writer.writerow(row)

# Create the CSV file if it does not exist
create_csv_file()
