"""Build a PDF report summarizing the four original-data experiments."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "experimental_results_report.pdf"


def report_table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F2F6F7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9AE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    return table


def add_figure(story: list, filename: str, caption: str, width: float = 17 * cm) -> None:
    path = RESULTS / filename
    if path.exists():
        image = Image(str(path))
        scale = min(width / image.imageWidth, 10.5 * cm / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        story.extend([image, Spacer(1, 0.15 * cm), Paragraph(caption, STYLES["Caption"])])


def page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#58676C"))
    canvas.drawString(2 * cm, 1.1 * cm, "NS_Implem: Original-Data Experimental Results")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(
    name="ReportTitle", parent=STYLES["Title"], fontName="Helvetica-Bold",
    fontSize=23, leading=28, textColor=colors.HexColor("#173B46"),
    alignment=TA_CENTER, spaceAfter=14,
))
STYLES.add(ParagraphStyle(
    name="Section", parent=STYLES["Heading1"], fontName="Helvetica-Bold",
    fontSize=16, leading=20, textColor=colors.HexColor("#1F4E5F"),
    spaceBefore=12, spaceAfter=8,
))
STYLES.add(ParagraphStyle(
    name="Subsection", parent=STYLES["Heading2"], fontName="Helvetica-Bold",
    fontSize=11, leading=14, textColor=colors.HexColor("#8A4B2A"),
    spaceBefore=8, spaceAfter=5,
))
STYLES.add(ParagraphStyle(
    name="BodyReport", parent=STYLES["BodyText"], fontSize=9.5, leading=14,
    textColor=colors.HexColor("#20292C"), spaceAfter=7,
))
STYLES.add(ParagraphStyle(
    name="Caption", parent=STYLES["BodyText"], fontSize=8, leading=10,
    alignment=TA_CENTER, textColor=colors.HexColor("#58676C"), spaceAfter=8,
))


def build() -> Path:
    body = STYLES["BodyReport"]
    story: list = [
        Spacer(1, 1.3 * cm),
        Paragraph("Representational Methods on Original Neuroscience Datasets", STYLES["ReportTitle"]),
        Paragraph("Experimental Results Report", STYLES["Title"]),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "This report collates four completed experiments using Haxby 2001, the Natural Scenes "
            "Dataset, and Pereira 2018. Kriegeskorte 2008 is retained as an original published "
            "reference RDM resource. Results are descriptive and should not be treated as "
            "population-level inferential claims.", body,
        ),
        Spacer(1, 0.4 * cm),
        report_table([
            ["Experiment", "Dataset", "Primary question"],
            ["1. Method agreement", "Haxby 2001", "Do representational estimators recover similar category geometry?"],
            ["2. Vision model vs brain", "NSD + ResNet-18", "Which model layer best matches image-level brain responses?"],
            ["3. Language model vs brain", "Pereira + DistilGPT-2", "Which transformer layer best matches language responses?"],
            ["4. Searchlight RSA", "Haxby 2001", "Where is object-category geometry spatially expressed?"],
        ], [3.4 * cm, 4.0 * cm, 9.0 * cm]),
        Paragraph("Executive Summary", STYLES["Section"]),
        Paragraph(
            "Haxby category geometry was reproducible across methods and localized strongly across "
            "independent run halves. ResNet-18 alignment with the three-session NSD subset was weak "
            "and metric-dependent. DistilGPT-2 produced measurable predictive alignment with Pereira "
            "responses, but all metrics favored the input embedding rather than a middle layer.", body,
        ),
        PageBreak(),
        Paragraph("1. Haxby Method Agreement", STYLES["Section"]),
        Paragraph(
            "Input: 12 runs x 8 object categories x 464 ventral-temporal voxels. The experiment "
            "compared classic correlation-distance RSA, crossnobis distances, a PCM-derived RDM, "
            "and an odd-run reference RDM using Spearman correlations of their upper triangles.", body,
        ),
        report_table([
            ["RDM comparison", "Spearman rho"],
            ["Classic RSA vs odd-run reference", "0.871"],
            ["Crossnobis vs PCM", "0.765"],
            ["Classic RSA vs PCM", "0.758"],
            ["Classic RSA vs crossnobis", "0.678"],
            ["PCM vs odd-run reference", "0.651"],
            ["Crossnobis vs odd-run reference", "0.517"],
        ], [10.5 * cm, 3.3 * cm]),
        Spacer(1, 0.2 * cm),
        Paragraph("Run-level results", STYLES["Subsection"]),
        Paragraph(
            "Mean CKA between runs was 0.393. Mean angular shape distance was 0.707 radians "
            "(about 40.5 degrees). Classic RSA generalized most strongly to the independent odd-run "
            "reference, while crossnobis and PCM agreed strongly with each other.", body,
        ),
    ]
    add_figure(story, "01_rdm_agreement.png", "Figure 1. Pairwise RDM agreement on Haxby subject 2.")
    story.extend([
        PageBreak(),
        Paragraph("2. ResNet-18 Versus NSD", STYLES["Section"]),
        Paragraph(
            "Input: 152 exactly matched natural images, three brain repetitions, 512 selected "
            "subject-1 voxels, and ImageNet-pretrained ResNet-18 features. Higher is better for RSA, "
            "CKA, and ridge correlation; lower is better for shape distance.", body,
        ),
        report_table([
            ["Layer", "Classic RSA", "Crossnobis", "CKA", "Shape dist.", "Ridge r"],
            ["conv1", "0.006", "0.001", "0.035", "1.436", "-0.018"],
            ["layer1", "-0.091", "-0.101", "0.065", "1.224", "0.000"],
            ["layer2", "-0.054", "-0.034", "0.074", "1.121", "0.014"],
            ["layer3", "-0.039", "0.002", "0.112", "1.014", "-0.008"],
            ["layer4", "-0.038", "-0.030", "0.162", "0.931", "-0.012"],
        ], [3.1 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 2.4 * cm, 2.1 * cm]),
        Paragraph("Interpretation", STYLES["Subsection"]),
        Paragraph(
            "Classic RSA favored conv1, crossnobis favored layer3, CKA and shape distance favored "
            "layer4, and ridge favored layer2. CKA and reversed shape distance had identical layer "
            "rankings, but the absolute effects were weak: RSA was near zero and the best ridge "
            "correlation was only 0.014. The normalized plot magnifies these small raw differences.", body,
        ),
    ])
    add_figure(story, "02_model_vs_brain.png", "Figure 2. Normalized metric profiles across ResNet-18 layers.")
    story.extend([
        PageBreak(),
        Paragraph("3. DistilGPT-2 Versus Pereira 2018", STYLES["Section"]),
        Paragraph(
            "Input: 384 sentences, 12,155 complete neuroids, and seven DistilGPT-2 representation "
            "stages. RSA compares sentence RDMs, CKA compares full representation spaces, and ridge "
            "predicts held-out neural responses.", body,
        ),
        report_table([
            ["Layer", "RSA", "CKA", "Ridge"],
            ["embedding", "0.042", "0.152", "0.232"],
            ["h1", "0.035", "0.150", "0.217"],
            ["h2", "-0.033", "0.042", "0.202"],
            ["h3", "-0.031", "0.045", "0.220"],
            ["h4", "-0.030", "0.056", "0.209"],
            ["h5", "0.009", "0.093", "0.212"],
            ["h6", "0.034", "0.049", "0.216"],
        ], [5.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]),
        Paragraph("Interpretation", STYLES["Subsection"]),
        Paragraph(
            "All methods selected the input embedding. RSA and CKA rankings agreed strongly "
            "(Spearman rho = 0.893). Ridge scores ranged from 0.202 to 0.232, but RSA and CKA were "
            "low. The expected middle-layer peak was not reproduced; architecture, sentence pooling, "
            "and multi-subject response construction are plausible contributors.", body,
        ),
    ])
    add_figure(story, "03_llm_layerwise.png", "Figure 3. Normalized Pereira alignment across DistilGPT-2 stages.")
    story.extend([
        PageBreak(),
        Paragraph("4. Haxby Searchlight RSA", STYLES["Section"]),
        Paragraph(
            "Odd runs defined an independent category RDM, while even runs supplied local spatial "
            "patterns. Each valid ventral-temporal searchlight was compared with the reference using "
            "Spearman correlation.", body,
        ),
        report_table([
            ["Statistic", "Result"],
            ["Map shape", "40 x 64 x 64"],
            ["Valid centers", "462"],
            ["Positive centers", "447 (96.8%)"],
            ["Median correlation", "0.313"],
            ["95th percentile", "0.616"],
            ["Peak voxel", "(30, 21, 29)"],
            ["Native scanner coordinate", "(-36.75, -39.38, -9.38) mm"],
            ["Peak correlation", "0.751"],
        ], [8.0 * cm, 6.0 * cm]),
        Paragraph("Interpretation", STYLES["Subsection"]),
        Paragraph(
            "Most valid searchlights positively reproduced the independent reference geometry. "
            "The peak coordinate is in native scanner space, not MNI space. The result supports "
            "spatially localized and repeatable object-category structure within this subject's "
            "ventral-temporal ROI.", body,
        ),
    ])
    add_figure(story, "04_searchlight.png", "Figure 4. Orthogonal slices through the searchlight peak.")
    story.extend([
        PageBreak(),
        Paragraph("Overall Conclusions", STYLES["Section"]),
        Paragraph(
            "The strongest evidence is the reproducibility and localization of Haxby category "
            "geometry. Pereira responses are predictably related to DistilGPT-2 representations, "
            "although the input embedding is unexpectedly strongest. The current NSD subset shows "
            "little absolute ResNet-brain alignment and substantial metric dependence.", body,
        ),
        Paragraph("Limitations", STYLES["Section"]),
        Paragraph(
            "No permutation tests, bootstrap confidence intervals, noise ceilings, group-level "
            "subject analyses, or corrected searchlight inference were calculated. Ridge results "
            "would benefit from nested hyperparameter selection. NSD uses one subject and only three "
            "sessions. Haxby is also a single-subject analysis. Accordingly, these results demonstrate "
            "the methods and generate hypotheses rather than establish population-level effects.", body,
        ),
        Paragraph("Recommended Next Steps", STYLES["Section"]),
        Paragraph(
            "Add stimulus-label permutation tests for RSA and CKA; bootstrap stimuli and subjects; "
            "estimate split-half noise ceilings; run additional NSD sessions and subjects; use nested "
            "cross-validation for ridge; and perform group-level, multiple-comparison-corrected "
            "searchlight inference.", body,
        ),
    ])

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm,
        topMargin=1.7 * cm, bottomMargin=1.8 * cm,
        title="Original-Data Experimental Results Report",
        author="NS_Implem",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())