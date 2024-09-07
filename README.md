# Intaligent_door_bell
## Face Recognition with Multi-Model Verification System  This project implements a real-time face recognition system using multiple face similarity models (FaceNet, ArcFace, MobileFaceNet) to improve accuracy and robustness. By combining the outputs of different models, the system verifies the identity of a person and minimizes false positives. The system can be used for various purposes such as access control, smart home systems, or security applications. -
## Features - 
**Multi-Model Verification**: Uses multiple models (FaceNet, ArcFace, MobileFaceNet) to verify the similarity of detected faces. - 
**Real-Time Face Recognition**: Captures video feed using a webcam and processes frames in real-time. - 
**Embeddings Persistence**: Face embeddings are saved and loaded for faster system initialization and efficient processing. - 
**YOLOv5 for Face Detection**: Uses YOLOv5 for real-time face detection, ensuring accuracy and speed. - 
**L2 Normalization**: All face embeddings are L2 normalized for better cosine similarity comparisons. - 
**Threshold-Based Similarity Check**: Adjustable thresholds for similarity checks, allowing flexible control of face matching sensitivity.  

## How It Works
1. **Face Detection**: The system detects faces in a live webcam feed using the YOLOv5 model. 
2. **Face Embeddings**: For each detected face, embeddings are generated using multiple models (FaceNet, ArcFace, and MobileFaceNet).
3. **Multi-Model Verification**: The embeddings are compared against a stored reference set using cosine similarity, and the system uses a majority voting system to confirm the identity of the person.
4. **Real-Time Feedback**: If the system recognizes the person, it labels them as a family member. Otherwise, the person is labeled as unknown.

## Setup and Installation 
1. Clone the repository:
   ```bash    git clone https://github.com/your-username/face-recognition-multi-model.git    cd face-recognition-multi-model
