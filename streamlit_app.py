# ============================================================
# STREAMLIT APP
# PNEUMONIA DETECTION SYSTEM FROM X-RAY IMAGES
# ============================================================
#
# WORKFLOW
#
# Upload image
#       ↓
# Color image?
#       ↓
# YES → Reject
#       ↓
# NO → Grayscale
#       ↓
# Modality Classifier
#       ↓
# ┌──────────────┬──────────────┬──────────────┐
# │    X-RAY     │      CT      │     MRI      │
# └──────┬───────┴──────┬───────┴──────┬───────┘
#        │              │              │
#        ↓              ↓              ↓
# Pneumonia Model     Reject         Reject
#        │
#   ┌────┴─────┐
#   ↓          ↓
# Normal   Pneumonia
#        ↓
#    PDF Report
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import io
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩻",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .normal-box {
        background-color: #eaf7ea;
    }

    .pneumonia-box {
        background-color: #fdeaea;
    }

    .modality-box {
        background-color: #eef4ff;
    }

    .rejected-box {
        background-color: #fff1f1;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Pneumonia Detection System from X-ray Images'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Automated medical image modality verification and '
    'pneumonia detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL FILES
# ============================================================

MODALITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "modality_classifier.keras"
)

PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# MODEL CLASS MAPPINGS
# ============================================================
#
# IMPORTANT:
#
# This mapping assumes your modality dataset was created as:
#
# 0_X-ray
# 1_CT
# 2_MRI
#
# Therefore:
#
# 0 = X-ray
# 1 = CT
# 2 = MRI
#
# ============================================================

MODALITY_LABELS = {
    0: "X-ray",
    1: "CT",
    2: "MRI"
}


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# This assumes your pneumonia dataset was:
#
# NORMAL
# PNEUMONIA
#
# Keras alphabetical mapping:
#
# NORMAL = 0
# PNEUMONIA = 1
#
# ============================================================

PNEUMONIA_LABELS = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"""
Modality model was not found.

Expected file:
{MODALITY_MODEL_PATH}

Required:
modality_classifier.keras
"""
        )

    model = tf.keras.models.load_model(
        MODALITY_MODEL_PATH,
        compile=False
    )

    return model


@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"""
Pneumonia model was not found.

Expected file:
{PNEUMONIA_MODEL_PATH}

Required:
best_xception_pneumonia_model.keras
"""
        )

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# GET MODEL INPUT SIZE
# ============================================================

def get_model_input_size(model):

    try:

        input_shape = model.input_shape

        if isinstance(
            input_shape,
            list
        ):

            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if (
            height is not None
            and width is not None
        ):

            return (
                int(height),
                int(width)
            )

    except Exception:
        pass

    # Fallback
    return (
        224,
        224
    )


# ============================================================
# GRAYSCALE CHECK
# ============================================================

def is_grayscale_image(image):
    """
    Determine whether the uploaded image is grayscale.

    Returns:
        True  -> grayscale
        False -> color
    """

    # --------------------------------------------------------
    # Native grayscale modes
    # --------------------------------------------------------

    if image.mode in [
        "1",
        "L",
        "I",
        "F",
        "I;16"
    ]:

        return True

    # --------------------------------------------------------
    # Convert other modes to RGB
    # --------------------------------------------------------

    rgb_image = image.convert(
        "RGB"
    )

    image_array = np.asarray(
        rgb_image
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if (
        image_array.ndim != 3
        or image_array.shape[-1] != 3
    ):

        return True

    # --------------------------------------------------------
    # Check RGB equality
    #
    # A grayscale image stored as RGB has:
    #
    # R = G = B
    #
    # --------------------------------------------------------

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    return (
        np.array_equal(
            red,
            green
        )
        and
        np.array_equal(
            green,
            blue
        )
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    target_size
):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    grayscale = image.convert(
        "L"
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    grayscale = grayscale.resize(
        target_size,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        grayscale
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # Convert grayscale → 3 channels
    #
    # This matches:
    #
    # input_shape=(224,224,3)
    #
    # --------------------------------------------------------

    image_array = np.stack(
        [
            image_array,
            image_array,
            image_array
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    model,
    image
):

    target_size = get_model_input_size(
        model
    )

    processed_image = preprocess_image(
        image,
        target_size
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    ).reshape(-1)

    # --------------------------------------------------------
    # Require exactly 3 outputs
    # --------------------------------------------------------

    if len(prediction) != 3:

        raise ValueError(
            f"""
Invalid modality model output.

Expected:
3 outputs

X-ray / CT / MRI

Received:
{len(prediction)} outputs
"""
        )

    # --------------------------------------------------------
    # Softmax normalization if needed
    # --------------------------------------------------------

    if (
        np.any(prediction < 0)
        or
        np.any(prediction > 1)
        or
        not np.isclose(
            np.sum(prediction),
            1.0,
            atol=1e-3
        )
    ):

        prediction = tf.nn.softmax(
            prediction
        ).numpy()

    # --------------------------------------------------------
    # Highest probability
    # --------------------------------------------------------

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_index]
    )

    modality = MODALITY_LABELS.get(
        class_index,
        "Unknown"
    )

    return (
        modality,
        confidence,
        prediction
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    model,
    image
):

    target_size = get_model_input_size(
        model
    )

    processed_image = preprocess_image(
        image,
        target_size
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    ).reshape(-1)

    # ========================================================
    # CASE 1 — TWO-CLASS SOFTMAX
    # ========================================================

    if len(prediction) == 2:

        # ----------------------------------------------------
        # Normalize if required
        # ----------------------------------------------------

        if (
            np.any(prediction < 0)
            or
            np.any(prediction > 1)
            or
            not np.isclose(
                np.sum(prediction),
                1.0,
                atol=1e-3
            )
        ):

            prediction = tf.nn.softmax(
                prediction
            ).numpy()

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_index]
        )

        diagnosis = PNEUMONIA_LABELS[
            class_index
        ]

        return (
            diagnosis,
            confidence,
            prediction
        )

    # ========================================================
    # CASE 2 — SINGLE SIGMOID OUTPUT
    # ========================================================

    if len(prediction) == 1:

        probability = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if (
            probability < 0
            or probability > 1
        ):

            probability = float(
                tf.sigmoid(
                    prediction[0]
                ).numpy()
            )

        # ----------------------------------------------------
        # 0.5 threshold
        # ----------------------------------------------------

        if probability >= 0.5:

            diagnosis = "Pneumonia"

            confidence = probability

        else:

            diagnosis = "Normal"

            confidence = (
                1.0 - probability
            )

        return (
            diagnosis,
            confidence,
            probability
        )

    # ========================================================
    # INVALID OUTPUT
    # ========================================================

    raise ValueError(
        f"""
Invalid pneumonia model output.

Expected:
2-class softmax OR
1-class sigmoid

Received:
{len(prediction)} outputs
"""
    )


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    modality,
    modality_confidence,
    diagnosis=None,
    diagnosis_confidence=None
):

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    result_style = ParagraphStyle(
        "ReportResult",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=16,
        leading=20,
        spaceAfter=10
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Pneumonia Detection System from X-ray Images",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Medical Image Analysis Report",
            subtitle_style
        )
    )

    # ========================================================
    # DATE
    # ========================================================

    report_date = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Analysis Information",
            heading_style
        )
    )

    report_data = [
        [
            "Analysis Date",
            report_date
        ],
        [
            "Image Type",
            "Grayscale"
        ],
        [
            "Detected Modality",
            modality
        ],
        [
            "Modality Confidence",
            f"{modality_confidence * 100:.2f}%"
        ]
    ]

    if diagnosis is not None:

        report_data.append(
            [
                "Final Result",
                diagnosis
            ]
        )

        report_data.append(
            [
                "Diagnosis Confidence",
                f"{diagnosis_confidence * 100:.2f}%"
            ]
        )

    report_table = Table(
        report_data,
        colWidths=[
            55 * mm,
            105 * mm
        ]
    )

    report_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    story.append(
        report_table
    )

    story.append(
        Spacer(1, 12)
    )

    # ========================================================
    # IMAGE
    # ========================================================

    story.append(
        Paragraph(
            "Analyzed Image",
            heading_style
        )
    )

    image_buffer = io.BytesIO()

    image.convert(
        "RGB"
    ).save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)

    report_image = ReportLabImage(
        image_buffer,
        width=100 * mm,
        height=100 * mm
    )

    story.append(
        report_image
    )

    story.append(
        Spacer(1, 12)
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if diagnosis is not None:

        story.append(
            Paragraph(
                "Final Pneumonia Analysis",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"<b>{diagnosis}</b>",
                result_style
            )
        )

        story.append(
            Paragraph(
                f"Model confidence: "
                f"{diagnosis_confidence * 100:.2f}%",
                ParagraphStyle(
                    "Confidence",
                    parent=normal_style,
                    alignment=TA_CENTER,
                    spaceAfter=15
                )
            )
        )

    else:

        story.append(
            Paragraph(
                "Pneumonia analysis was not performed because "
                "the detected image modality was not X-ray.",
                normal_style
            )
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Research Disclaimer:</b> This application is "
            "intended for research and decision-support purposes. "
            "The result should not be considered a medical diagnosis "
            "and should be reviewed by a qualified healthcare "
            "professional.",
            normal_style
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.markdown(
        """
        **Processing pipeline**

        1. Image upload
        2. Grayscale verification
        3. X-ray / CT / MRI classification
        4. X-ray → pneumonia detection
        5. Normal / Pneumonia result
        6. PDF report
        """
    )

    st.divider()

    st.subheader(
        "Required Models"
    )

    st.code(
        """
modality_classifier.keras

best_xception_pneumonia_model.keras
        """,
        language="text"
    )

    st.divider()

    st.caption(
        "Research / decision-support application"
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ],
    help=(
        "Upload a grayscale medical image. "
        "Color images will be rejected."
    )
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image = Image.open(
            uploaded_file
        )

        image.load()

    except Exception as error:

        st.error(
            "Unable to read the uploaded image."
        )

        st.code(
            str(error)
        )

        st.stop()

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Uploaded Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded medical image",
        width="stretch"
    )

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Image mode:** `{image.mode}`"
        )

    with col2:

        st.write(
            f"**Resolution:** "
            f"`{image.width} × {image.height}`"
        )

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "Analyze",
        type="primary",
        width="stretch"
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        # ====================================================
        # STEP 1
        # COLOR IMAGE CHECK
        # ====================================================

        if not is_grayscale_image(
            image
        ):

            st.markdown(
                '<div class="result-box rejected-box">',
                unsafe_allow_html=True
            )

            st.error(
                "Image Rejected"
            )

            st.write(
                "The uploaded image is a color image."
            )

            st.write(
                "This system accepts grayscale medical images only."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.stop()

        # ====================================================
        # GRAYSCALE ACCEPTED
        # ====================================================

        st.success(
            "Grayscale image detected."
        )

        # ====================================================
        # STEP 2
        # LOAD MODALITY MODEL
        # ====================================================

        with st.spinner(
            "Loading medical image modality classifier..."
        ):

            try:

                modality_model = (
                    load_modality_model()
                )

            except Exception as error:

                st.error(
                    "Modality classifier could not be loaded."
                )

                st.code(
                    str(error)
                )

                st.stop()

        # ====================================================
        # STEP 3
        # MODALITY PREDICTION
        # ====================================================

        with st.spinner(
            "Identifying X-ray, CT, or MRI..."
        ):

            try:

                (
                    modality,
                    modality_confidence,
                    modality_probabilities
                ) = predict_modality(
                    modality_model,
                    image
                )

            except Exception as error:

                st.error(
                    "Medical image modality detection failed."
                )

                st.code(
                    str(error)
                )

                st.stop()

        # ====================================================
        # MODALITY RESULT
        # ====================================================

        st.markdown(
            '<div class="result-box modality-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Detected Image Type"
        )

        st.write(
            f"## {modality}"
        )

        st.write(
            f"Confidence: "
            f"**{modality_confidence * 100:.2f}%**"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # ====================================================
        # MODALITY PROBABILITIES
        # ====================================================

        with st.expander(
            "View modality probabilities"
        ):

            st.write(
                f"X-ray: "
                f"{modality_probabilities[0] * 100:.2f}%"
            )

            st.progress(
                float(modality_probabilities[0])
            )

            st.write(
                f"CT: "
                f"{modality_probabilities[1] * 100:.2f}%"
            )

            st.progress(
                float(modality_probabilities[1])
            )

            st.write(
                f"MRI: "
                f"{modality_probabilities[2] * 100:.2f}%"
            )

            st.progress(
                float(modality_probabilities[2])
            )

        # ====================================================
        # STEP 4
        # CT
        # ====================================================

        if modality == "CT":

            st.warning(
                "CT image detected."
            )

            st.info(
                "The pneumonia model is designed for "
                "chest X-ray images. Pneumonia analysis "
                "was therefore not performed."
            )

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence
            )

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="medical_image_report.pdf",
                mime="application/pdf",
                width="stretch"
            )

            st.stop()

        # ====================================================
        # STEP 5
        # MRI
        # ====================================================

        if modality == "MRI":

            st.warning(
                "MRI image detected."
            )

            st.info(
                "The pneumonia model is designed for "
                "chest X-ray images. Pneumonia analysis "
                "was therefore not performed."
            )

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence
            )

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="medical_image_report.pdf",
                mime="application/pdf",
                width="stretch"
            )

            st.stop()

        # ====================================================
        # STEP 6
        # X-RAY
        # ====================================================

        if modality == "X-ray":

            st.success(
                "X-ray detected. Proceeding to pneumonia detection."
            )

            # =================================================
            # LOAD PNEUMONIA MODEL
            # =================================================

            with st.spinner(
                "Loading proposed pneumonia model..."
            ):

                try:

                    pneumonia_model = (
                        load_pneumonia_model()
                    )

                except Exception as error:

                    st.error(
                        "Pneumonia model could not be loaded."
                    )

                    st.code(
                        str(error)
                    )

                    st.stop()

            # =================================================
            # PNEUMONIA PREDICTION
            # =================================================

            with st.spinner(
                "Analyzing X-ray for pneumonia..."
            ):

                try:

                    (
                        diagnosis,
                        diagnosis_confidence,
                        pneumonia_probabilities
                    ) = predict_pneumonia(
                        pneumonia_model,
                        image
                    )

                except Exception as error:

                    st.error(
                        "Pneumonia detection failed."
                    )

                    st.code(
                        str(error)
                    )

                    st.stop()

            # =================================================
            # FINAL RESULT
            # =================================================

            if diagnosis == "Pneumonia":

                st.markdown(
                    '<div class="result-box pneumonia-box">',
                    unsafe_allow_html=True
                )

                st.error(
                    "Pneumonia Detected"
                )

                st.write(
                    f"Confidence: "
                    f"**{diagnosis_confidence * 100:.2f}%**"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-box normal-box">',
                    unsafe_allow_html=True
                )

                st.success(
                    "Normal"
                )

                st.write(
                    f"Confidence: "
                    f"**{diagnosis_confidence * 100:.2f}%**"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            # =================================================
            # PNEUMONIA PROBABILITIES
            # =================================================

            if (
                isinstance(
                    pneumonia_probabilities,
                    np.ndarray
                )
                and
                pneumonia_probabilities.size == 2
            ):

                with st.expander(
                    "View pneumonia probabilities"
                ):

                    st.write(
                        f"Normal: "
                        f"{pneumonia_probabilities[0] * 100:.2f}%"
                    )

                    st.progress(
                        float(
                            pneumonia_probabilities[0]
                        )
                    )

                    st.write(
                        f"Pneumonia: "
                        f"{pneumonia_probabilities[1] * 100:.2f}%"
                    )

                    st.progress(
                        float(
                            pneumonia_probabilities[1]
                        )
                    )

            # =================================================
            # PDF REPORT
            # =================================================

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence,
                diagnosis=diagnosis,
                diagnosis_confidence=diagnosis_confidence
            )

            st.divider()

            st.subheader(
                "Final Report"
            )

            st.download_button(
                "Download Final Report (PDF)",
                data=pdf_bytes,
                file_name="pneumonia_detection_report.pdf",
                mime="application/pdf",
                width="stretch"
            )
