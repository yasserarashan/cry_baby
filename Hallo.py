import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from io import BytesIO
import tensorflow as tf
import numpy as np
import librosa
import subprocess

# تحميل النموذج عند بدء التشغيل
MODEL_PATH = "improved_model_all_end.h5"
try:
    print("🔄 جاري تحميل النموذج من:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ تم تحميل النموذج بنجاح!")
except Exception as e:
    print("❌ خطأ في تحميل النموذج:", e)
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
        print("🔄 بدء تحويل الملف إلى WAV...")
        temp_wav = BytesIO()
        if ext == "caf":
            subprocess.run(['ffmpeg', '-i', '-', '-f', 'wav', '-'], input=audio_bytes, stdout=temp_wav, check=True)
        else:
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format=ext)
            audio.export(temp_wav, format="wav")
        temp_wav.seek(0)
        print("✅ تم تحويل الملف إلى WAV بنجاح")
        return temp_wav
    except Exception as e:
        print("❌ خطأ أثناء تحويل الملف إلى WAV:", e)
        raise HTTPException(status_code=400, detail="فشل تحويل الملف إلى WAV")

# دالة لاستخراج الميزات من الصوت
def extract_features(audio_io: BytesIO, frame_duration: float = 2.0):
    try:
        y, sr = librosa.load(audio_io, mono=True)
        frame_samples = int(frame_duration * sr)
        if len(y) < frame_samples:
            raise HTTPException(status_code=400, detail="ملف الصوت قصير جدًا")
        segment = y[:frame_samples]
        features = np.array([
            np.mean(librosa.feature.chroma_stft(y=segment, sr=sr)),
            np.mean(librosa.feature.rms(y=segment)),
            np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=segment))
        ])
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في استخراج الميزات: {e}")

@app.get("/")
def health_check():
    return {"message": "تم تحميل النموذج"}

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    ext = file.filename.rsplit('.', 1)[1].lower()
    audio_bytes = await file.read()
    convert_to_wav(audio_bytes, ext)
    return {"message": "تم تحويل الصوت إلى WAV"}

@app.post("/extract")
async def extract_audio_features(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    ext = file.filename.rsplit('.', 1)[1].lower()
    audio_bytes = await file.read()
    wav_audio = convert_to_wav(audio_bytes, ext)
    features = extract_features(wav_audio)
    return {"message": "تم استخراج الميزات", "features": features.tolist()}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    ext = file.filename.rsplit('.', 1)[1].lower()
    audio_bytes = await file.read()
    wav_audio = convert_to_wav(audio_bytes, ext)
    features = extract_features(wav_audio)
    input_data = np.expand_dims(features, axis=0)
    predictions = model.predict(input_data)[0]
    predicted_class = np.argmax(predictions)
    probabilities = {f"Class {i}": float(pred) for i, pred in enumerate(predictions)}
    return {
        "message": "تم التصنيف بنجاح",
        "predicted_class": int(predicted_class),
        "probabilities": probabilities
    }

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
