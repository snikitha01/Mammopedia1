from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wdbc_service import predict_wdbc


app = FastAPI(
    title="Mammopedia API",
    description="Breast cancer analysis backend",
    version="1.0.0"
)


# Allow the Lovable frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input format for WDBC analysis
class WDBCRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="30 WDBC diagnostic features in the original dataset order"
    )


# Basic endpoints
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


# WDBC prediction endpoint
@app.post("/analyze/tabular")
def analyze_tabular(request: WDBCRequest):
    try:
        result = predict_wdbc(request.features)

        return {
            "success": True,
            "dataset": "WDBC",
            "model": "Logistic Regression",
            "result": result
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )
