from fastapi import FastAPI

app = FastAPI(
    title="Mammopedia API",
    description="Breast cancer analysis backend",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "application": "Mammopedia",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
