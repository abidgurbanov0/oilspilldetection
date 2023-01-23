from tkinter.tix import Form
from fastapi import FastAPI, File, UploadFile,  Depends
from fastapi.security.api_key import  APIKey
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import auth
from oilspill  import imagetaketaker
from fastapi import FastAPI, Form



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/upload")
async def upload( text: str = Form()):

    res = imagetaketaker(text)    

    return {"message": res}
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
       
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    a=file.filename
    res=imagetaketaker(a)
    return {"message": res}

if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
