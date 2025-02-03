import cv2
import os
import sys

# Initialize the Haar cascade algorithm
alg = "./haarcascade_frontalface_default.xml"
if not os.path.exists(alg):
    print(f"Error: file not found at {alg}")
    sys.exit(1)
haar_cascade = cv2.CascadeClassifier(alg)

# define folder paths
folder_dir = "./pre-process-faces" # path to pre-processed images 
output_dir = "./stored-faces" # extracted faces folder 

# check if input folder exists 
if not os.path.exists(folder_dir):
    print(f"Error: Image folder {folder_dir} does not exist")
    sys.exit(1) 

# get image filenames from directory 
image_files = [
    os.path.join(folder_dir, img) for img in os.listdir(folder_dir)
    if img.lower().endswith((".jpg", ".png", "jpeg"))
]

if not image_files:
    print("Error: no iamges found in directory")
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Created directory 'stored-faces'")

print("Checkpoint: Pre-face extraction")

# Function to extract and save faces from an image
def extract_faces(image_path, output_dir, start_index=0):
    print(f"Processing: {image_path}")
    
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)
    
    if img is None:
        print(f"Warning: Failed to load {image_path}")
        return 0

    # Detect faces in the image
    # note: this may need to be adjusted based on the image quality for improved accuracy 
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")

    # Save each detected face
    for i, (x, y, w, h) in enumerate(faces, start=start_index):
        cropped_face = img[y : y + h, x : x + w]
        target_file_name = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(target_file_name, cropped_face)
    
    # Return the number of faces detected for unique naming across images
    return len(faces)

# Process each image and save the faces
face_index = 0
for image_file in image_files:
    face_index += extract_faces(image_file, output_dir, start_index=face_index)

print("Face extraction process completed")