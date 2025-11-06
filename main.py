# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import face_recognition
import cv2
import numpy as np
import pickle
import os
import base64
from datetime import datetime
import io
from PIL import Image

app = FastAPI()

# Enable CORS for your Lovable app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
ENCODINGS_FILE = "face_encodings.pkl"
DATASET_PATH = "dataset"

# Ensure dataset directory exists
os.makedirs(DATASET_PATH, exist_ok=True)

# Load encodings at startup
known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
        print(f"Loaded {len(known_face_encodings)} face encodings")
    else:
        print("No encodings file found. Starting fresh.")

load_encodings()

# Models
class CaptureResponse(BaseModel):
    success: bool
    message: str
    images_captured: int

class TrainResponse(BaseModel):
    success: bool
    message: str
    total_faces: int

class AttendanceRequest(BaseModel):
    image: str  # base64 encoded image

class AttendanceResponse(BaseModel):
    success: bool
    recognized: bool
    roll_no: Optional[str] = None
    confidence: Optional[float] = None
    message: str

@app.get("/")
def read_root():
    return {
        "message": "Face Recognition API for Attendance System",
        "endpoints": {
            "POST /capture-face": "Capture face images for a student",
            "POST /train-model": "Train/retrain the face recognition model",
            "POST /recognize-face": "Recognize face and return roll number",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "encodings_loaded": len(known_face_encodings),
        "students_enrolled": len(set(known_face_names))
    }

@app.post("/capture-face", response_model=CaptureResponse)
async def capture_face(
    roll_no: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Capture and save face images for a student.
    Expects roll_no and multiple image files.
    """
    try:
        # Create student directory
        student_dir = os.path.join(DATASET_PATH, roll_no)
        os.makedirs(student_dir, exist_ok=True)
        
        images_saved = 0
        for idx, image_file in enumerate(images):
            # Read image
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect face using face_recognition library
            face_locations = face_recognition.face_locations(img)
            
            if len(face_locations) > 0:
                # Save the image
                img_path = os.path.join(student_dir, f"{idx + 1}.jpg")
                cv2.imwrite(img_path, img)
                images_saved += 1
        
        return CaptureResponse(
            success=True,
            message=f"Successfully captured {images_saved} face images for {roll_no}",
            images_captured=images_saved
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model", response_model=TrainResponse)
async def train_model():
    """
    Train/retrain the face recognition model with all images in the dataset.
    """
    global known_face_encodings, known_face_names
    
    try:
        new_encodings = []
        new_names = []
        
        # Loop through each student directory
        for roll_no in os.listdir(DATASET_PATH):
            student_dir = os.path.join(DATASET_PATH, roll_no)
            
            if not os.path.isdir(student_dir):
                continue
            
            # Loop through each image
            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                
                try:
                    # Load and encode the image
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        new_encodings.append(encodings[0])
                        new_names.append(roll_no)
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Save the new encodings
        data = {"encodings": new_encodings, "names": new_names}
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        # Update global variables
        known_face_encodings = new_encodings
        known_face_names = new_names
        
        return TrainResponse(
            success=True,
            message=f"Model trained successfully with {len(new_encodings)} face encodings",
            total_faces=len(new_encodings)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize-face", response_model=AttendanceResponse)
async def recognize_face(image_file: UploadFile = File(...)):
    """
    Recognize a face from an uploaded image and return the roll number.
    """
    try:
        if len(known_face_encodings) == 0:
            return AttendanceResponse(
                success=False,
                recognized=False,
                message="No trained faces in the system. Please train the model first."
            )
        
        # Read and decode the image
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 0:
            return AttendanceResponse(
                success=True,
                recognized=False,
                message="No face detected in the image"
            )
        
        # Check the first face found
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                return AttendanceResponse(
                    success=True,
                    recognized=True,
                    roll_no=roll_no,
                    confidence=float(confidence),
                    message=f"Face recognized: {roll_no}"
                )
        
        return AttendanceResponse(
            success=True,
            recognized=False,
            message="Face not recognized in the system"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-student/{roll_no}")
async def remove_student(roll_no: str):
    """
    Remove a student's face data and retrain the model.
    """
    try:
        student_dir = os.path.join(DATASET_PATH, roll_no)
        
        if os.path.exists(student_dir):
            # Remove the directory
            import shutil
            shutil.rmtree(student_dir)
            
            # Retrain the model
            await train_model()
            
            return {
                "success": True,
                "message": f"Student {roll_no} removed and model retrained"
            }
        else:
            return {
                "success": False,
                "message": f"Student {roll_no} not found"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
