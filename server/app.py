from fastapi import FastAPI
from server import app as main_app

app = main_app

@app.get("/")
def home():
    return {"message": "Your app is working"}
