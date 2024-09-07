import os
import torch
import cv2
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from numpy.linalg import norm
import pickle  # Import pickle to save and load embeddings

# Load YOLOv5 model for face detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load FaceNet model for face recognition
embedder = FaceNet()

# Path to family images for generating embeddings
family_image_dir = './croped_images'

# Path to save the family member embeddings
embedding_save_path = './family_member_embeddings.pkl'

# Function to load an image and convert it to embeddings
def get_face_embeddings(image):
    if isinstance(image, str):  # If it's a file path
        img = Image.open(image)
        img = np.array(img)
    else:  # If it's a numpy array
        img = image
    
    img = cv2.resize(img, (160, 160))  # Resize face to 160x160 as required by FaceNet
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    embeddings = embedder.embeddings(img)
    return embeddings[0]

# Generate family member embeddings
def generate_family_member_embeddings(family_dir):
    family_embeddings = {}
    
    # Loop over each family member's directory
    for family_member in os.listdir(family_dir):
        member_dir = os.path.join(family_dir, family_member)
        if not os.path.isdir(member_dir):
            continue

        embeddings_list = []
        
        # Loop over each image in the family member's folder
        for image_name in os.listdir(member_dir):
            image_path = os.path.join(member_dir, image_name)
            embeddings = get_face_embeddings(image_path)
            embeddings_list.append(embeddings)

        # Average embeddings for each family member
        avg_embedding = np.mean(embeddings_list, axis=0)
        family_embeddings[family_member] = avg_embedding
        print(f"Generated embedding for {family_member}")

    return family_embeddings

# Function to save embeddings to a file
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

# Function to load embeddings from a file
def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded from {file_path}")
        return embeddings
    else:
        print(f"No embeddings file found at {file_path}, generating new embeddings...")
        return generate_family_member_embeddings(family_image_dir)

# Function to detect if the face is a family member
def is_family_member(face_embeddings, threshold=0.6):
    for name, family_embedding in family_member_embeddings.items():
        similarity = np.dot(face_embeddings, family_embedding) / (norm(face_embeddings) * norm(family_embedding))
        if similarity > threshold:
            return True, name
    return False, None

# Real-time face recognition using webcam
def detect_and_recognize_webcam():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects (including faces) using YOLOv5
        results = model(frame)
        
        # Convert results to pandas dataframe for easy access
        df = results.pandas().xyxy[0]

        # Filter to process only detected "person" objects
        persons = df[df['name'] == 'person']

        for _, person in persons.iterrows():
            # Extract face bounding box
            x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
            face_img = frame[y1:y2, x1:x2]

            # Get face embeddings using FaceNet
            embeddings = get_face_embeddings(face_img)

            # Check if face is a family member
            is_family, name = is_family_member(embeddings)

            if is_family:
                label = f"Family: {name}"
                print(f"Family member detected: {name}")
            else:
                label = "Unknown"
                print("Unknown person detected!")

            # Draw the bounding box and label
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow("Real-time Face Recognition", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Load family member embeddings (from file if exists, otherwise generate)
family_member_embeddings = load_embeddings(embedding_save_path)

# If new embeddings are generated, save them to a file
if not os.path.exists(embedding_save_path):
    save_embeddings(family_member_embeddings, embedding_save_path)

# Start real-time recognition
detect_and_recognize_webcam()
