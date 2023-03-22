from fastapi import APIRouter
from src.modules.model007 import model007


router = APIRouter()
router.include_router(model007.router)

 