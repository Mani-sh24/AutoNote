from fastapi import FastAPI , File , UploadFile
# from faster_whisper import WhisperModel
from processing import summarise_extractive
import warnings
import mlx_whisper
import os 

os.makedirs("contents" , exist_ok=True)
warnings.filterwarnings("ignore")
# --- ORIGINAL STARTUP MODEL LOADING ---
# model  = WhisperModel("tiny"  ,  device="cpu",compute_type="int8",cpu_threads=4) // not suitable for apple silicon
# model = WhisperModel("tiny", device="auto", compute_type="int8", cpu_threads=8) // not suitable for apple silicon
model = mlx_whisper.load_models.load_model("mlx-community/whisper-small-mlx")

# --- ALTERNATIVE STARTUP MODEL OPTIONS (Comment/Uncomment to switch) ---
model = mlx_whisper.load_models.load_model("mlx-community/whisper-base-mlx") # Tiny (~75MB) - lighter and faster for 8GB RAM
# model = None # Set to None if you do not want to load any model at server startup (saves memory)

# --- ACTIVE TRANSCRIPTION MODEL (Used in the /upload-audio endpoint) ---
# Choose which model to use during transcription. Uncomment one of the lines below:
# ACTIVE_WHISPER_MODEL = "mlx-community/whisper-tiny-mlx"   # Tiny (~75MB) - Recommended for 8GB M1 (Fastest, low accuracy)
ACTIVE_WHISPER_MODEL = "mlx-community/whisper-base-mlx"   # Base (~140MB) - Basic accuracy
# ACTIVE_WHISPER_MODEL = "mlx-community/whisper-small-mlx"  # Small (~480MB) - Balanced speed and accuracy
# ACTIVE_WHISPER_MODEL = "mlx-community/whisper-medium-mlx" # Medium (~1.5GB) - Higher accuracy, slower

app = FastAPI()


def save_transcription(content):
    with open("txrt1.txt", "w", encoding="utf-8") as transcript_file:
        transcript_file.write(content)


@app.get("/")
def hello():
    return {"hello":1}

@app.post("/upload-audio")
async def read_audio(file: UploadFile):
    if file.content_type != "audio/mpeg":
        return {"Error":"Error only audio files are supported"}
    if file.size and file.size > 100 * 1024 * 1024:
        return {"error": "File too large"}
    
    fpath = os.path.join("contents", file.filename)
    contents = await file.read()
    with open(fpath, "wb") as f:
        f.write(contents)
    await file.close()
    
    result = mlx_whisper.transcribe(fpath, path_or_hf_repo=ACTIVE_WHISPER_MODEL)
    res = result["text"]    
    os.remove(fpath)
    save_transcription(res)
    summary = summarise_extractive(res)
    return {"filename": file.filename, "file-type" :file.content_type,  "filesize": file.size, "Summary": summary}

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

# this route uses faster-whisper i am using mlx-whisper cuz apple silicon faster results
# @app.post("/upload-audio")
# async def read_audio(file:UploadFile):
#     if file.size and file.size > 100 * 1024 * 1024:
#      return {"error": "File too large"}
#     segments , info = model.transcribe(file.file)
#     await file.close()
#     res = ""
#     for seg in segments:
#         res = res+seg.text
#     del segments
#     del info

#     sumry = summarise_extractive(res)
#     return {"filename": file.filename , "filesize" :file.size  , "Summary" : sumry}
