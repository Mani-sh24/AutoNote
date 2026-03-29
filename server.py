from fastapi import FastAPI , File , UploadFile
# from faster_whisper import WhisperModel
from processing import *
import warnings
import mlx_whisper
import os 

os.makedirs("contents" , exist_ok=True)

warnings.filterwarnings("ignore")
# model  = WhisperModel("tiny"  ,  device="cpu",compute_type="int8",cpu_threads=4) // not suitable for apple silicon
# model = WhisperModel("tiny", device="auto", compute_type="int8", cpu_threads=8) // not suitable for apple silicon
model = mlx_whisper.load_models.load_model("mlx-community/whisper-small-mlx")
app = FastAPI()
@app.get("/")
def hello():
    return {"hello":1}

@app.post("/upload-audio")
async def read_audio(file: UploadFile):
    if file.size and file.size > 100 * 1024 * 1024:
        return {"error": "File too large"}
    
    fpath = os.path.join("contents", file.filename)
    contents = await file.read()
    with open(fpath, "wb") as f:
        f.write(contents)
    await file.close()
    
    result = mlx_whisper.transcribe(fpath, path_or_hf_repo="mlx-community/whisper-tiny-mlx")
    res = result["text"]    
    os.remove(fpath)

    summary = summarise_extractive(res)
    return {"filename": file.filename, "filesize": file.size, "Summary": summary}

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

def summarise_extractive(content):
    cleaned_text = cleantext(content)
    tokens, sentences = process_text(cleaned_text)
    word_frequency = wordFreq(tokens, sentences)
    sentence_scores = sent_score(sentences, word_frequency)
    select_len = max(ceil(len(sentences) * SUMMARY_LEN), 5)
    summary = nlargest(select_len, sentence_scores, key=sentence_scores.get)
    res = " ".join(sent.text for sent in sorted(summary, key=lambda s: s.start))
    return res

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