import os
import numpy as np
import librosa
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import joblib
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import shutil

# إعدادات TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# تهيئة التطبيق
app = FastAPI()

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# التهيئات الأساسية
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}

# تحميل النموذج
try:
    model = tf.keras.models.load_model('improved_model_all_end.h5')
    scaler = joblib.load('scaler_cry_all_end.pkl')
    label_encoder = joblib.load('label_encoder_all_end.pkl')
except Exception as e:
    raise RuntimeError(f"خطء في تحميل النموذج: {str(e)}")

# الدوال المساعدة
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                raise HTTPException(400, detail="فشل تحويل الملف إلى WAV")
    return file_path

def extract_features(file_path: str, frame_duration: float = 2.0) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, mono=True)
    except Exception as e:
        raise HTTPException(400, detail=f"خطء في تحميل الصوت: {str(e)}")
    
    frame_samples = int(frame_duration * sr)
    if len(y) < frame_samples:
        raise HTTPException(400, detail="مدة الصوت قصيرة جدًا")
    
    features_list = []
    for i in range(len(y) // frame_samples):
        start = i * frame_samples
        end = start + frame_samples
        segment = y[start:end]
        
        features = [
            np.mean(librosa.feature.chroma_stft(y=segment, sr=sr)),
            np.mean(librosa.feature.rms(y=segment)),
            np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=segment))
        ]
        
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20), axis=1)
        features.extend(mfccs.tolist())
        features_list.append(features)
    
    return np.array(features_list)

# النقاط الطرفية
@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "OK", "docs": "/docs"}

@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("favicon.ico")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="نوع الملف غير مدعوم")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        # حفظ الملف
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # معالجة الصوت
        wav_path = convert_to_wav(file_path)
        features = extract_features(wav_path)
        
        # التنبؤ
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        predictions = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(np.mean(predictions, axis=0))])[0]
        
        return {"prediction": predicted_label}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        # تنظيف الملفات المؤقتة
        for path in [file_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
