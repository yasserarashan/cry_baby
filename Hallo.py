import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
import subprocess
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# إعدادات TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# تحميل النموذج والمحول والمُشفِّر
try:
    model = tf.keras.models.load_model('improved_model_all_end.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # تجميع النموذج لإزالة التحذير
    scaler = joblib.load('scaler_cry_all_end.pkl')
    label_encoder = joblib.load('label_encoder_all_end.pkl')
except Exception as e:
    raise RuntimeError(f"خطأ في تحميل النموذج أو البيانات: {str(e)}")

# الدوال المساعدة
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(audio_bytes: bytes, ext: str) -> BytesIO:
    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format=ext)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except CouldntDecodeError:
        raise HTTPException(400, detail="فشل تحويل الملف إلى WAV")
    except Exception:
        raise HTTPException(400, detail="تنسيق الملف غير مدعوم")

def extract_features(audio_io: BytesIO, frame_duration: float = 2.0) -> np.ndarray:
    try:
        y, sr = librosa.load(audio_io, mono=True)
    except Exception as e:
        raise HTTPException(400, detail=f"خطأ في تحميل الصوت: {str(e)}")
    
    frame_samples = int(frame_duration * sr)
    if len(y) < frame_samples:
        raise HTTPException(400, detail="مدة الصوت قصيرة جدًا")
    
    features_list = []
    for i in range(len(y) // frame_samples):
        segment = y[i * frame_samples : (i + 1) * frame_samples]
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

# نقطة نهاية التحقق من الصحة
@app.get("/")
async def health_check():
    return {"status": "OK", "docs": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="نوع الملف غير مدعوم")
    
    try:
        ext = file.filename.rsplit('.', 1)[-1].lower()
        audio_bytes = await file.read()
        wav_io = convert_to_wav(audio_bytes, ext)
        features = extract_features(wav_io)
        
        # تطبيع الميزات
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        
        # إجراء التوقع باستخدام النموذج
        predictions = model.predict(features_scaled)
        
        # حساب متوسط الاحتمالات لكل فئة عبر جميع الإطارات
        avg_prediction = np.mean(predictions, axis=0)
        
        # الحصول على الفئة ذات أعلى احتمال
        predicted_index = np.argmax(avg_prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        # إرجاع النتيجة
        probabilities = {label_encoder.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(avg_prediction)}
        
        return {
            "prediction": predicted_label,
            "probabilities": probabilities
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"خطأ داخلي: {str(e)}")

# تشغيل التطبيق
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
