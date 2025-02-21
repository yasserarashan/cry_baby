import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from io import BytesIO
import tempfile
import subprocess
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

# إعدادات المسارات الخاصة بالملفات المطلوبة
MODEL_PATH = "improved_model_all_end.h5"
SCALER_PATH = "scaler_cry_all_end.pkl"
ENCODER_PATH = "label_encoder_all_end.pkl"

# تحميل النموذج والمعاملات
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"خطأ في تحميل النموذج أو الملفات: {str(e)}")

# إعداد التطبيق
app = FastAPI()

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قائمة الامتدادات المدعومة
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'caf', '3gp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# دالة لتحويل الملف إلى WAV إذا لم يكن بصيغة WAV
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
                raise HTTPException(400, detail="خطأ في تحويل الملف إلى WAV")
    return file_path

# دالة استخراج الميزات من الملف الصوتي
def extract_features(file_path: str, frame_duration: float = 2.0) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, mono=True)
    except Exception as e:
        raise HTTPException(400, detail=f"خطأ في تحميل الملف الصوتي: {e}")

    frame_samples = int(frame_duration * sr)
    num_frames = len(y) // frame_samples
    if num_frames == 0:
        raise HTTPException(400, detail="ملف الصوت قصير جدًا.")

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

@app.get("/")
async def health_check():
    return {"status": "OK", "docs": "/docs"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="نوع الملف غير مدعوم")
    
    # حفظ الملف المرفوع مؤقتًا
    try:
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(500, detail=f"خطأ في حفظ الملف: {e}")
    
    try:
        # تحويل الملف إلى WAV إن لزم الأمر
        wav_path = convert_to_wav(tmp_path)
        
        # استخراج الميزات من الملف الصوتي
        features = extract_features(wav_path)
        if features.size == 0:
            raise HTTPException(400, detail="لم يتم استخراج أي ميزات من الملف.")
        
        # تطبيع الميزات وإعادة تشكيلها لتتناسب مع مدخلات النموذج
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        
        # إجراء التوقع باستخدام النموذج
        predictions = model.predict(features_scaled)
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        # إعداد احتمالات التصنيف لكل فئة
        probabilities = {}
        for i, prob in enumerate(avg_prediction):
            label = label_encoder.inverse_transform([i])[0]
            probabilities[label] = float(prob)
        
        return {"prediction": predicted_label, "probabilities": probabilities}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"خطأ داخلي: {e}")
    finally:
        # حذف الملفات المؤقتة بعد الانتهاء
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if wav_path != tmp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
