from server import app
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Your app is working"}
