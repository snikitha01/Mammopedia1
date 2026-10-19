from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wdbc_service import predict_wdbc, explain_wdbc


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
@app.post("/explain/tabular")
def explain_tabular(request: WDBCRequest):
    try:
        explanation = explain_wdbc(request.features)
        return {
            "success": True,
            "dataset": "WDBC",
            "model": "Logistic Regression",
            "explanation": explanation,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Explanation generation failed.",
        )

@app.post("/report/generate")
def generate_report(request: WDBCRequest):
    """Generate a PDF report from a fresh WDBC prediction."""
    from io import BytesIO
    from datetime import datetime, timezone

    from fastapi.responses import StreamingResponse
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    try:
        result = predict_wdbc(request.features)

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        story = [
            Paragraph("MAMMOPEDIA", styles["Title"]),
            Paragraph(
                "WDBC Breast Cancer Research Report",
                styles["Heading2"],
            ),
            Paragraph(
                "Generated: "
                + datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                styles["Normal"],
            ),
            Spacer(1, 18),
            Paragraph("Analysis Results", styles["Heading2"]),
        ]

        details = [
            ["Dataset", "WDBC"],
            ["Model", "Logistic Regression"],
            ["Input feature count", str(len(request.features))],
            [
                "Prediction",
                str(result.get("diagnosis", "Unavailable")).title(),
            ],
            [
                "Malignant probability",
                f"{float(result.get('malignant_probability', 0)) * 100:.4f}%",
            ],
            [
                "Benign probability",
                f"{float(result.get('benign_probability', 0)) * 100:.4f}%",
            ],
        ]

        table = Table(details, colWidths=[5 * cm, 10 * cm])
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#E8EAF6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.extend([table, Spacer(1, 18)])
        story.append(
            Paragraph("Limitations and Safety", styles["Heading2"])
        )
        story.append(
            Paragraph(
                "This report contains a model prediction based only on "
                "the supplied WDBC numerical features. It does not include "
                "mammography inference, genomic analysis, or quantum-model "
                "inference. Model probabilities are not clinical risk "
                "estimates. This research prototype is not a medical "
                "diagnosis and must not replace professional medical care.",
                styles["BodyText"],
            )
        )

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    'attachment; filename="mammopedia_wdbc_report.pdf"'
            },
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PDF report generation failed.",
        )

