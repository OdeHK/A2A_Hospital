from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import Settings
from api.routers import users

setting = Settings()

app = FastAPI(
    title=setting.PROJECT_NAME,
    version=setting.VERSION,
)
app.include_router(users.router)
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI App"}
