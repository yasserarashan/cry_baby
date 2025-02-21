import os
import numpy as np
import librosa
import io
import tensorflow as tf
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# إعداد TensorFlow
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

# تحميل النموذج
try:
    print("🔄 جاري تحميل النموذج والملفات المساعدة...")
    model = tf.keras.models.load_model('improved_model_all_end.h5')
    model.compile()  # إصلاح مشكلة compile_metrics
    scaler = joblib.load('scaler_cry_all_end.pkl')
    label_encoder = joblib.load('label_encoder_all_end.pkl')
    print("✅ تم تحميل النموذج بنجاح!")
except Exception as e:
    raise RuntimeError(f"❌ خطأ في تحميل النموذج: {str(e)}")

# قائمة الامتدادات المدعومة
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}

def allowed_file(filename: str) -> bool:
    """التحقق من نوع الملف"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_bytes: bytes, file_ext: str) -> np.ndarray:
    """تحويل الصوت إلى WAV وتحميله في Librosa مباشرة"""
    try:
        # تحويل الصوت إلى صيغة مناسبة باستخدام `pydub`
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=file_ext)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # تحميل الصوت باستخدام Librosa
        y, sr = librosa.load(wav_io, mono=True)
        return y, sr
    except CouldntDecodeError:
        raise HTTPException(400, detail="❌ فشل تحليل الملف الصوتي")
    except Exception as e:
        raise HTTPException(400, detail=f"❌ خطأ في تحويل الصوت: {str(e)}")

def extract_features(y: np.ndarray, sr: int, frame_duration: float = 2.0) -> np.ndarray:
    """استخراج الميزات الصوتية من البيانات"""
    frame_samples = int(frame_duration * sr)
    if len(y) < frame_samples:
        raise HTTPException(400, detail="❌ مدة الصوت قصيرة جدًا")

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

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "OK", "docs": "/docs"}

@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("favicon.ico")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """استقبال ملف صوتي وإجراء التنبؤ"""
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="❌ نوع الملف غير مدعوم")

    try:
        # قراءة الملف في الذاكرة
        file_bytes = await file.read()
        file_ext = file.filename.rsplit('.', 1)[1].lower()

        # تحويل الصوت إلى WAV وتحميله
        y, sr = convert_to_wav(file_bytes, file_ext)

        # استخراج الميزات
        features = extract_features(y, sr)

        # إجراء التنبؤ
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        predictions = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(np.mean(predictions, axis=0))])[0]

        return {"prediction": predicted_label}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"❌ خطأ داخلي: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
