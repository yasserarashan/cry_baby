import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from io import BytesIO
import tensorflow as tf
import numpy as np
import librosa

# تحميل النموذج عند بدء التشغيل
MODEL_PATH = "improved_model_all_end.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"خطأ في تحميل النموذج: {e}")

# تهيئة التطبيق وإضافة CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# قائمة الامتدادات المسموح بها
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "caf", "3gp"}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# دالة لتحويل الملف الصوتي إلى WAV
def convert_to_wav(audio_bytes: bytes, ext: str) -> BytesIO:
    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format=ext)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception:
        raise HTTPException(status_code=400, detail="فشل تحويل الملف إلى WAV")

# دالة لاستخراج الميزات من الصوت (مثال بسيط على استخراج الميزات من أول 2 ثانية)
def extract_features(audio_io: BytesIO, frame_duration: float = 2.0):
    try:
        y, sr = librosa.load(audio_io, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في تحميل الصوت: {e}")
    
    frame_samples = int(frame_duration * sr)
    if len(y) < frame_samples:
        raise HTTPException(status_code=400, detail="ملف الصوت قصير جدًا")
    
    segment = y[:frame_samples]
    features = {
        "chroma_stft": float(np.mean(librosa.feature.chroma_stft(y=segment, sr=sr))),
        "rms": float(np.mean(librosa.feature.rms(y=segment))),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
    }
    return features

@app.get("/")
def health_check():
    return {"message": "تم تحميل النموذج"}

# نقطة رفع الملف التي تقوم بتحويل الصوت إلى WAV واستخراج الميزات
@app.post("/extract")
async def extract_audio_features(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
         raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    audio_bytes = await file.read()
    wav_audio = convert_to_wav(audio_bytes, ext)
    
    features = extract_features(wav_audio)
    return {"message": "تم استخراج الميزات", "features": features}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
