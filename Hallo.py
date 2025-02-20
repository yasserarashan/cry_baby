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
# إعداد التطبيق
app = FastAPI()

# تفعيل CORS للسماح بالطلبات من الواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلد التخزين
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# أنواع الملفات المسموحة
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}

# تحميل النموذج والمعاملات
model = tf.keras.models.load_model('improved_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# التحقق من نوع الملف
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# تحويل الصوت إلى WAV إذا لزم الأمر
def convert_to_wav(file_path: str) -> str:
    ext = file_path.split('.')[-1].lower()
    if ext != 'wav':
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        try:
            sound = AudioSegment.from_file(file_path, format=ext)
            sound.export(wav_path, format="wav")
            return wav_path
        except CouldntDecodeError:
            try:
                subprocess.run(['ffmpeg', '-i', file_path, wav_path, '-y'], check=True)
                return wav_path
            except subprocess.CalledProcessError:
                raise HTTPException(status_code=400, detail="Error converting file to WAV")
    return file_path

# استخراج الميزات من الصوت
def extract_features(file_path: str, frame_duration: float = 2.0) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {e}")
    
    frame_samples = int(frame_duration * sr)
    num_frames = len(y) // frame_samples
    if num_frames == 0:
        raise HTTPException(status_code=400, detail="Audio file is too short.")
    
    features_list = []
    for i in range(num_frames):
        start, end = i * frame_samples, (i + 1) * frame_samples
        segment = y[start:end]
        features = [
            np.mean(librosa.feature.chroma_stft(y=segment, sr=sr)),
            np.mean(librosa.feature.rms(y=segment)),
            np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=segment))
        ]
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1).tolist()
        features.extend(mfccs)
        features_list.append(features)
    
    return np.array(features_list)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    try:
        wav_path = convert_to_wav(file_path)
        features = extract_features(wav_path)
        if features.size == 0:
            raise HTTPException(status_code=400, detail="No features extracted.")
        
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        predictions = model.predict(features_scaled)
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        os.remove(file_path)
        if wav_path != file_path:
            os.remove(wav_path)
        
        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

