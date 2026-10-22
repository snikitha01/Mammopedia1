from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from quantum_service import run_qsvm_experiment
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
    """Generate a PDF report with WDBC prediction and feature contributions."""
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
        KeepTogether,
    )
    from reportlab.graphics.shapes import Drawing, String, Rect
    from reportlab.lib.colors import HexColor

    try:
        result = predict_wdbc(request.features)
        explanation_result = explain_wdbc(request.features)
        explained_features = explanation_result.get("features", [])

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=1.7 * cm,
            bottomMargin=1.7 * cm,
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
                + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
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
                ("BACKGROUND", (0, 0), (0, -1), HexColor("#E8EAF6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.extend([table, Spacer(1, 18)])

        story.append(
            Paragraph("Feature Contributions", styles["Heading2"])
        )
        story.append(
            Paragraph(
                "The following features have the largest absolute signed "
                "score contributions for this prediction.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))

        # Sort by absolute contribution and show the 10 strongest.
        top_features = sorted(
            explained_features,
            key=lambda item: abs(float(item.get("contribution", 0))),
            reverse=True,
        )[:10]

        if top_features:
            max_value = max(
                abs(float(item["contribution"])) for item in top_features
            ) or 1.0

            chart_width = 450
            chart_height = 28 * len(top_features) + 20
            drawing = Drawing(chart_width, chart_height)

            for index, item in enumerate(top_features):
                y = chart_height - (index + 1) * 28
                name = str(item.get("feature", "Unknown feature"))
                contribution = float(item.get("contribution", 0))
                direction = str(item.get("direction", ""))
                bar_width = abs(contribution) / max_value * 175
                is_malignant = direction == "toward_malignant"
                bar_color = (
                    HexColor("#D9485F")
                    if is_malignant
                    else HexColor("#159570")
                )

                drawing.add(
                    String(0, y + 7, name[:25], fontSize=8)
                )
                drawing.add(
                    Rect(
                        170, y + 4, bar_width, 10,
                        fillColor=bar_color,
                        strokeColor=None,
                    )
                )
                drawing.add(
                    String(
                        355, y + 6,
                        f"{contribution:+.3f}",
                        fontSize=8,
                    )
                )

            story.append(drawing)
            story.append(Spacer(1, 8))

            contribution_rows = [
                ["Feature", "Contribution", "Direction"]
            ]
            for item in top_features:
                contribution = float(item["contribution"])
                direction = item.get("direction", "")
                label = (
                    "Toward malignant"
                    if direction == "toward_malignant"
                    else "Toward benign"
                    if direction == "toward_benign"
                    else "Unspecified"
                )
                contribution_rows.append([
                    str(item.get("feature", "Unknown feature")),
                    f"{contribution:+.4f}",
                    label,
                ])

            contribution_table = Table(
                contribution_rows,
                colWidths=[6 * cm, 3 * cm, 5 * cm],
                repeatRows=1,
            )
            contribution_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E8EAF6")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ])
            )
            story.extend([
                contribution_table,
                Spacer(1, 12),
            ])
        else:
            story.append(
                Paragraph(
                    "Feature contributions were not available for this analysis.",
                    styles["BodyText"],
                )
            )

        story.append(
            Paragraph("Interpretation and Limitations", styles["Heading2"])
        )
        story.append(
            Paragraph(
                "Contributions are signed Logistic Regression score "
                "contributions. Positive values move the model score toward "
                "malignant; negative values move it toward benign. These "
                "values are not causal effects or clinical risk estimates.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))
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

@app.post("/quantum/train")
def train_quantum_model():
    try:
        return run_qsvm_experiment()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"QSVM experiment failed: {str(exc)}",
        )
