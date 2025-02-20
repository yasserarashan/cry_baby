import os
import numpy as np
import librosa
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import shutil

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}
PORT = int(os.getenv("PORT", 9000))  # Get port from environment variable

# Load model and preprocessing objects
try:
    model = tf.keras.models.load_model('improved_model_all_end.h5')
    scaler = joblib.load('scaler_cry_all_end.pkl')
    label_encoder = joblib.load('label_encoder_all_end.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessing objects: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Service is running"}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path: str) -> str:
    # ... (ابقى نفس الدالة كما هي)

def extract_features(file_path: str, frame_duration: float = 2.0) -> np.ndarray:
    # ... (ابقى نفس الدالة كما هي)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not allowed_file(file.filename):
            raise HTTPException(400, detail="Unsupported file format")
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        wav_path = convert_to_wav(file_path)
        features = extract_features(wav_path)
        
        if features.size == 0:
            raise HTTPException(400, detail="Audio too short for analysis")
        
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        predictions = model.predict(features_scaled)
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        # Cleanup
        for path in [file_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        return {"prediction": predicted_label}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
