from fastapi import APIRouter, UploadFile
import tensorflow as tf
import numpy as np
import os

#APIRouter creates path operations for model007 module
router = APIRouter(
    prefix="/model007",
    tags=["model007"],
    responses={404: {"description": "Not found"}},
)

@router.post("/prediction")
async def prediction(file: UploadFile):
    UPLOAD_DIR = "src/modules/model007/uploads"
    if not os.path.exists(UPLOAD_DIR):
        os.mkdir(UPLOAD_DIR)

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    model = tf.keras.models.load_model('src/modules/model007/rakan.model')
    prediction = model.predict([file])
    return {
        "result": prediction
    }
