import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from io import BytesIO
import tensorflow as tf

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

# نقطة فحص الخدمة تُظهر أن النموذج تم تحميله
@app.get("/")
def health_check():
    return {"message": "تم تحميل النموذج"}

# نقطة رفع الملف التي تحول الصوت إلى WAV
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    audio_bytes = await file.read()
    wav_audio = convert_to_wav(audio_bytes, ext)
    
    # يمكنك استخدام wav_audio في عمليات أخرى إذا رغبت
    return {"message": "تم تحويل الصوت وتم تحميل النموذج"}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
