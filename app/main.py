from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api_v1.api import router as api_router
import uvicorn

# app = FastAPI(root_path='/prod')
app = FastAPI()

origins = ["http://localhost", "http://159.138.45.180"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8080, log_level="info", reload=True)
