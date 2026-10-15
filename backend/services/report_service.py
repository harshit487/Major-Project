"""
Stage 10: PDF Prediction Report Generation Service using ReportLab.

Generates a clean, academic/clinical-style downloadable PDF report containing:
- University & Team Header (Graphic Era Hill University, Team CSE27-304)
- Prediction ID & Timestamp
- Predicted Class (Benign / Malignant)
- Benign Probability, Malignant Probability, Confidence Score, Decision Threshold
- Original Uploaded Dermoscopic Image
- Grad-CAM Heatmap & Blended Overlay Visualizations (when available)
- Model Architecture & Preprocessing Specification
- Prominent Medical Safety Disclaimer (explicitly stating this is NOT a confirmed
  clinical diagnosis)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.config import (
    INPUT_SHAPE,
    MEDICAL_DISCLAIMER,
    MODEL_ARCHITECTURE,
    PROJECT_METADATA,
    REPORTS_DIR,
)


def generate_prediction_pdf_report(
    prediction_data: Dict[str, Any],
    output_dir: Path = REPORTS_DIR,
    original_image_path: Optional[Path] = None,
    heatmap_image_path: Optional[Path] = None,
    overlay_image_path: Optional[Path] = None,
) -> Path:
    """
    Create a formatted PDF report for a skin lesion classification result and
    return the path to the generated `.pdf` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_id = str(prediction_data.get("prediction_id", "pred_report"))
    safe_id = "".join(c for c in prediction_id if c.isalnum() or c in ("_", "-")) or "pred_report"
    pdf_path = output_dir / f"skin_lesion_report_{safe_id}.pdf"

    # Extract internal paths if attached in cache
    internal_paths = prediction_data.get("_internal_paths", {})
    orig_path = original_image_path or internal_paths.get("original_path")
    heat_path = heatmap_image_path or internal_paths.get("heatmap_path")
    over_path = overlay_image_path or internal_paths.get("overlay_path")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"Skin Lesion AI Analysis Report - {safe_id}",
        author="Graphic Era Hill University - Team CSE27-304",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#7f1d1d"),
    )

    story = []

    # 1. Institutional & Project Header
    story.append(Paragraph(PROJECT_METADATA["title"], title_style))
    story.append(
        Paragraph(
            f"<b>{PROJECT_METADATA['institution']}</b> | {PROJECT_METADATA['department']}<br/>"
            f"Project Team ID: <b>{PROJECT_METADATA['team_id']}</b> | "
            f"Faculty Guide: {PROJECT_METADATA['faculty_guide']}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # 2. Prediction Summary Table
    story.append(Paragraph("1. Prediction Summary (Experimental Research Output)", section_heading))

    predicted_class = str(prediction_data.get("predicted_class", "Unknown"))
    benign_prob = float(prediction_data.get("benign_probability", 0.0))
    malignant_prob = float(prediction_data.get("malignant_probability", 0.0))
    confidence_score = float(prediction_data.get("confidence_score", max(benign_prob, malignant_prob)))
    threshold = float(prediction_data.get("decision_threshold", 0.50))
    timestamp = str(
        prediction_data.get(
            "timestamp",
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
    )

    class_color = "#dc2626" if predicted_class.lower() == "malignant" else "#0d9488"

    summary_rows = [
        [
            Paragraph("<b>Prediction ID</b>", body_style),
            Paragraph(safe_id, body_style),
            Paragraph("<b>Timestamp (UTC)</b>", body_style),
            Paragraph(timestamp[:19].replace("T", " "), body_style),
        ],
        [
            Paragraph("<b>Predicted Category</b>", body_style),
            Paragraph(
                f"<font color='{class_color}'><b>{predicted_class.upper()}</b></font> "
                f"({'Cancerous / Neoplastic Risk' if predicted_class.lower() == 'malignant' else 'Non-Cancerous'})",
                body_style,
            ),
            Paragraph("<b>Decision Threshold</b>", body_style),
            Paragraph(f"{threshold:.2f} (Malignant if P >= {threshold:.2f})", body_style),
        ],
        [
            Paragraph("<b>Benign Probability</b>", body_style),
            Paragraph(f"{benign_prob * 100:.2f}% ({benign_prob:.4f})", body_style),
            Paragraph("<b>Malignant Probability</b>", body_style),
            Paragraph(f"{malignant_prob * 100:.2f}% ({malignant_prob:.4f})", body_style),
        ],
        [
            Paragraph("<b>Estimated Confidence</b>", body_style),
            Paragraph(f"{confidence_score * 100:.2f}% (Uncalibrated)", body_style),
            Paragraph("<b>Output Status</b>", body_style),
            Paragraph("Experimental Research Estimate (Not a Clinical Diagnosis)", body_style),
        ],
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[1.45 * inch, 2.15 * inch, 1.45 * inch, 2.15 * inch],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 3. Dermoscopic Image & Grad-CAM Visual Explainability
    story.append(Paragraph("2. Dermoscopic Image & Grad-CAM Visual Explainability", section_heading))

    img_cells = []
    captions = []
    for label_text, path_candidate in [
        ("Original Dermoscopic Input (224x224)", orig_path),
        ("Grad-CAM Activation Heatmap", heat_path),
        ("Blended Grad-CAM Overlay", over_path),
    ]:
        if path_candidate and Path(path_candidate).exists():
            img_cells.append(RLImage(str(path_candidate), width=2.1 * inch, height=2.1 * inch))
            captions.append(Paragraph(f"<b>{label_text}</b>", body_style))

    if img_cells:
        col_w = 7.2 * inch / len(img_cells)
        img_table = Table([img_cells, captions], colWidths=[col_w] * len(img_cells))
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(img_table)
    else:
        story.append(
            Paragraph(
                "<i>Visual image artifacts were not attached to this report request.</i>",
                body_style,
            )
        )

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Explainable AI (Grad-CAM) Interpretation Note:</b> Grad-CAM highlights the spatial regions "
            "in the final convolutional feature map of EfficientNetB0 that contributed most strongly to the "
            "model's prediction score. Highlighted regions indicate where the neural network focused; they "
            "do NOT prove histological malignancy or establish a medical diagnosis.",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    # 4. Model Information Table
    story.append(Paragraph("3. Deep Learning Model Specification", section_heading))
    model_rows = [
        [
            Paragraph("<b>Primary Backbone</b>", body_style),
            Paragraph(f"{MODEL_ARCHITECTURE} (Transfer Learning)", body_style),
            Paragraph("<b>Input Resolution</b>", body_style),
            Paragraph(f"{INPUT_SHAPE[0]} x {INPUT_SHAPE[1]} x {INPUT_SHAPE[2]} RGB", body_style),
        ],
        [
            Paragraph("<b>Training Dataset</b>", body_style),
            Paragraph("HAM10000 (7-to-2 Binary Mapping)", body_style),
            Paragraph("<b>Classification Head</b>", body_style),
            Paragraph("GlobalAvgPool2D -> Dropout -> Sigmoid", body_style),
        ],
        [
            Paragraph("<b>Benign Mapping (0)</b>", body_style),
            Paragraph("nv, bkl, df, vasc", body_style),
            Paragraph("<b>Malignant Mapping (1)</b>", body_style),
            Paragraph("mel, bcc, akiec", body_style),
        ],
    ]
    model_table = Table(
        model_rows,
        colWidths=[1.45 * inch, 2.15 * inch, 1.45 * inch, 2.15 * inch],
    )
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(model_table)
    story.append(Spacer(1, 14))

    # 5. Mandatory Medical Disclaimer Box
    disclaimer_box = Table(
        [
            [
                Paragraph(
                    f"<b>MANDATORY MEDICAL & RESEARCH DISCLAIMER</b><br/>{MEDICAL_DISCLAIMER}",
                    disclaimer_style,
                )
            ]
        ],
        colWidths=[7.2 * inch],
    )
    disclaimer_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#dc2626")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(disclaimer_box)

    doc.build(story)
    return pdf_path
