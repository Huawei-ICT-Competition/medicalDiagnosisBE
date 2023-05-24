from fastapi import APIRouter

from .endpoints import classify_img

router = APIRouter()
router.include_router(classify_img.router, prefix="/classify_img")
