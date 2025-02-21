import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# إعدادات التطبيق
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

@app.get("/")
async def health_check():
    return {"status": "OK", "docs": "/docs"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="نوع الملف غير مدعوم")
    
    return {"message": "تم الاتصال بالسيرفر بنجاح"}

# تشغيل التطبيق
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
