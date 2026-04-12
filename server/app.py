from fastapi import FastAPI
from server import app as main_app

app = main_app

@app.get("/")
def home():
    return {"message": "Your app is working"}


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
