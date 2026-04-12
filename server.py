import uvicorn
from openenv.core.env_server import create_fastapi_app

from insurance import InsuranceEnvironment, InsuranceAction, InsuranceObservation

app = create_fastapi_app(
    InsuranceEnvironment,   
    InsuranceAction,
    InsuranceObservation,
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
