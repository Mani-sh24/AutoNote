from fastapi import FastAPI , File , UploadFile
from faster_whisper import WhisperModel
import warnings

import os 

os.makedirs("contents" , exist_ok=True)

warnings.filterwarnings("ignore")
model  = WhisperModel("tiny"  ,  device="cpu",compute_type="int8",cpu_threads=4)
app = FastAPI()
@app.get("/")
def hello():
    return {"hello":1}

@app.post("/upload-audio")
async def read_audio(file:UploadFile):
    if file.size and file.size > 100 * 1024 * 1024:
        return {"error": "File too large"}
    
    segments , info = model.transcribe(file.file)
    await file.close()
    res = ""

    for seg in segments:
        res = res+seg.text
    del segments
    del info
    with open("Trans.txt" , "w+") as f:
        f.write(res)
    return {"filename": file.filename , "filesize" :file.size , "filetype":file.content_type , "Transcription":res}

@app.post("/test")
async def test_f(file:UploadFile = File(...)):
    print({        "filename":file.filename,
        "filetype":file.content_type})
    read_bytes = await file.read()
    fpath = os.path.join("contents" , file.filename)
    with open(fpath , "wb+") as f:
        f.write(read_bytes)
    return {
        "filename":file.filename,
        "filetype":file.content_type
    }