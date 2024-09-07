import os
import cv2
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

# Paths
input_image_dir = './input_images'  # Directory where original images will be saved
preprocessed_image_dir = './preprocessed_images'  # Directory to save preprocessed images
augmented_image_dir = './augmented_images'  # Directory to save augmented images
croped_image_dir = './croped_images' #  Directory to save croped images

# Ensure output directories exist
os.makedirs(preprocessed_image_dir, exist_ok=True)
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(croped_image_dir, exist_ok=True)

# Data Augmentation Setup
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),  # Rotation
    iaa.LinearContrast((0.8, 1.2)),  # Contrast adjustment
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Multiply((0.8, 1.2))  # Brightness adjustment
])

def capture_images(person_name, num_images=10):
    """Capture images using webcam and save them in a folder for the given person."""

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    person_dir = os.path.join(input_image_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"Capturing {num_images} images for {person_name}. Press 'Spacebar' to capture and save image, 'q' to quit.")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        save_frame = frame.copy()
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around detected faces
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face
        
        # Display the frame with rectangles around faces
        cv2.imshow(f"Capturing {person_name}", frame)

        # Check if the spacebar is pressed to save the image
        key = cv2.waitKey(1)
        if key == 32:  # 32 is the ASCII code for the spacebar
            if len(faces) > 0:  # Save the image only if faces are detected
                image_path = os.path.join(person_dir, f"{person_name}_{count + 1}.jpg")
                cv2.imwrite(image_path, save_frame)
                count += 1
                print(f"Captured image {count} for {person_name}")

        # Press 'q' to quit early
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing images for {person_name}")

def preprocess_image(image_path, output_path):
    """Preprocess the image by resizing and saving."""
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB mode
    img = img.resize((160, 160))  # Resize image to 160x160
    img.save(output_path)

def augment_image(image_path, output_path):
    """Apply augmentation to the image and save it."""
    img = cv2.imread(image_path)
    img_aug = aug_seq(image=img)
    cv2.imwrite(output_path, img_aug)

def crop_face(image_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        cv2.imwrite(output_path, face)

def process_images(input_dir, preprocessed_dir, augmented_dir, croped_dir):
    """Process images: preprocess, augment, and save them."""
    for person_folder in os.listdir(input_dir):
        person_folder_path = os.path.join(input_dir, person_folder)
        if not os.path.isdir(person_folder_path):
            continue
        
        # Create directories for preprocessed and augmented images
        person_preprocessed_dir = os.path.join(preprocessed_dir, person_folder)
        person_augmented_dir = os.path.join(augmented_dir, person_folder)
        person_croped_dir = os.path.join(croped_dir, person_folder)
        os.makedirs(person_preprocessed_dir, exist_ok=True)
        os.makedirs(person_augmented_dir, exist_ok=True)
        os.makedirs(person_croped_dir, exist_ok=True)

        # Process each image in the person's folder
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            
            # Preprocess image
            preprocessed_image_path = os.path.join(person_preprocessed_dir, image_name)
            preprocess_image(image_path, preprocessed_image_path)
            
            # Augment image
            augmented_image_path = os.path.join(person_augmented_dir, image_name)
            augment_image(preprocessed_image_path, augmented_image_path)

            # Crop image
            croped_image_path = os.path.join(person_croped_dir, image_name)
            crop_face(image_path, croped_image_path)

# Step 1: Capture images for multiple family members
family_members = ["Parson1","Person2","Person3"]  # Add names of family members here
for member in family_members:
    capture_images(person_name=member, num_images=20)  # Capture 10 images per member

# Step 2: Preprocess and augment captured images
process_images(input_image_dir, preprocessed_image_dir, augmented_image_dir, croped_image_dir)

print("Dataset capturing, preprocessing, and augmentation completed.")

