from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from ml.classifier import model
import numpy as np
import cv2

router = APIRouter()

@router.post('')
async def classify_image(file: UploadFile):
    nparr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return JSONResponse({'prediction': model.preprocess_predict(img)})
