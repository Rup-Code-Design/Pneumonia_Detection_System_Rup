# ============================================================
# PNEUMONIA DETECTION SYSTEM FROM X-RAY IMAGES
# ============================================================
#
# Workflow:
#   1. Upload image
#   2. Reject color images
#   3. Accept grayscale images
#   4. Identify modality:
#        0 = CT
#        1 = MRI
#        2 = X-ray
#   5. If X-ray -> proposed pneumonia model
#   6. Display Normal / Pneumonia
#   7. Generate downloadable PDF report
#
# ============================================================

import os
import io
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf

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
    Image as RLImage,
)


# ============================================================
# PAGE TITLE WITH LUNG X-RAY ICON
# ============================================================

ICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lung_xray_icon.png"
)

# ------------------------------------------------------------
# TITLE ROW
# ------------------------------------------------------------

title_col1, title_col2 = st.columns(
    [1, 6],
    vertical_alignment="center"
)

with title_col1:

    if os.path.exists(ICON_PATH):

        st.image(
            ICON_PATH,
            width=80
        )

    else:

        st.error(
            "lung_xray_icon.png not found"
        )


with title_col2:

    st.markdown(
        """
        <h1 style="
            font-size: 32px;
            font-weight: 600;
            line-height: 1.15;
            margin: 0;
            padding: 0;
        ">
            PNEUMONIA DETECTION SYSTEM<br>
            FROM X-RAY IMAGES
        </h1>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SUBTITLE
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 25px;
    ">
        Automated medical image modality verification and
        pneumonia detection
    </div>
    """,
    unsafe_allow_html=True
)
# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# 3-CLASS MODALITY CLASSIFIER
#
# IMPORTANT CLASS MAPPING:
#
#   0 = CT
#   1 = MRI
#   2 = X-ray
#
# ------------------------------------------------------------

MODALITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "modality_classifier.keras"
)


# ------------------------------------------------------------
# PROPOSED PNEUMONIA MODEL
# ------------------------------------------------------------

PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# CLASS LABELS
# ============================================================

# DO NOT CHANGE THIS ORDER
#
# Model output:
#   index 0 -> CT
#   index 1 -> MRI
#   index 2 -> X-ray

MODALITY_LABELS = [
    "CT",
    "MRI",
    "X-ray",
]


# ============================================================
# IMAGE SIZE
# ============================================================

DEFAULT_IMAGE_SIZE = (
    224,
    224
)


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
Modality classifier was not found.

Expected location:
{MODALITY_MODEL_PATH}

Required file:
modality_classifier.keras

Required class mapping:

0 = CT
1 = MRI
2 = X-ray
"""
        )

    try:

        model = tf.keras.models.load_model(
            MODALITY_MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "Could not load the X-ray/CT/MRI modality model.\n\n"
            f"Original error:\n{str(e)}"
        )


@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"""
Pneumonia model was not found.

Expected location:
{PNEUMONIA_MODEL_PATH}

Make sure the model file is committed to the same
GitHub repository as streamlit_app.py.
"""
        )

    try:

        # ----------------------------------------------------
        # Load complete .keras model directly
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "Could not load the proposed pneumonia model.\n\n"
            f"Model:\n{PNEUMONIA_MODEL_PATH}\n\n"
            f"Original error:\n{str(e)}"
        )


# ============================================================
# GET MODEL INPUT SIZE
# ============================================================

def get_model_input_size(model):

    try:

        shape = model.input_shape

        if isinstance(
            shape,
            list
        ):

            shape = shape[0]

        height = shape[1]
        width = shape[2]

        if (
            height is not None
            and
            width is not None
        ):

            return (
                int(height),
                int(width)
            )

    except Exception:

        pass

    return DEFAULT_IMAGE_SIZE


# ============================================================
# CHECK WHETHER IMAGE IS GRAYSCALE
# ============================================================

def is_grayscale_image(
    image: Image.Image
):

    """
    Returns:

        True  -> grayscale
        False -> color
    """

    # --------------------------------------------------------
    # Explicit grayscale PIL modes
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

    rgb = image.convert(
        "RGB"
    )

    arr = np.asarray(
        rgb
    )

    # --------------------------------------------------------
    # RGB image must have 3 channels
    # --------------------------------------------------------

    if (
        arr.ndim != 3
        or
        arr.shape[-1] != 3
    ):

        return True

    # --------------------------------------------------------
    # Check R = G = B
    # --------------------------------------------------------

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    return (
        np.array_equal(r, g)
        and
        np.array_equal(g, b)
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image,
    target_size
):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    gray = gray.resize(
        target_size,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    arr = np.asarray(
        gray
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize
    #
    # 0-255 -> 0-1
    # --------------------------------------------------------

    arr = arr / 255.0

    # --------------------------------------------------------
    # Grayscale -> 3 channels
    # --------------------------------------------------------

    arr = np.stack(
        [
            arr,
            arr,
            arr
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# ============================================================
# GENERIC MODEL PREDICTION
# ============================================================

def get_prediction_array(
    model,
    image_array
):

    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

    return prediction


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

    processed = preprocess_image(
        image,
        target_size
    )

    prediction = get_prediction_array(
        model,
        processed
    )

    # --------------------------------------------------------
    # Flatten output
    # --------------------------------------------------------

    prediction = prediction.reshape(
        -1
    )

    # ========================================================
    # THREE-CLASS SOFTMAX
    #
    # 0 = CT
    # 1 = MRI
    # 2 = X-ray
    # ========================================================

    if len(prediction) == 3:

        # ----------------------------------------------------
        # Apply softmax if values are not normalized
        # ----------------------------------------------------

        if not np.isclose(
            np.sum(prediction),
            1.0,
            atol=1e-3
        ):

            prediction = tf.nn.softmax(
                prediction
            ).numpy()

        # ----------------------------------------------------
        # Find predicted class
        # ----------------------------------------------------

        class_index = int(
            np.argmax(
                prediction
            )
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            prediction[class_index]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The same prediction array is returned.
        #
        # Index mapping:
        #
        # prediction[0] = CT
        # prediction[1] = MRI
        # prediction[2] = X-ray
        # ----------------------------------------------------

        return (
            MODALITY_LABELS[class_index],
            confidence,
            prediction
        )

    # ========================================================
    # BINARY OUTPUT
    # ========================================================

    if len(prediction) == 1:

        raise ValueError(
            """
The modality model has only one output.

The application requires a 3-class model:

0 = CT
1 = MRI
2 = X-ray

Your current model appears to be binary.
"""
        )

    # ========================================================
    # UNEXPECTED OUTPUT
    # ========================================================

    raise ValueError(
        f"""
Unexpected modality model output shape.

Received:
{prediction.shape}

Expected:
3 classes:

0 = CT
1 = MRI
2 = X-ray
"""
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

    processed = preprocess_image(
        image,
        target_size
    )

    prediction = get_prediction_array(
        model,
        processed
    )

    prediction = prediction.reshape(
        -1
    )

    # ========================================================
    # BINARY SIGMOID
    #
    # 0 = Normal
    # 1 = Pneumonia
    # ========================================================

    if len(prediction) == 1:

        probability = float(
            prediction[0]
        )

        if (
            probability < 0.0
            or
            probability > 1.0
        ):

            probability = float(
                tf.sigmoid(
                    prediction[0]
                ).numpy()
            )

        if probability >= 0.5:

            label = "Pneumonia"

            confidence = probability

        else:

            label = "Normal"

            confidence = (
                1.0 - probability
            )

        return (
            label,
            confidence,
            probability
        )

    # ========================================================
    # TWO-CLASS SOFTMAX
    #
    # 0 = Normal
    # 1 = Pneumonia
    # ========================================================

    if len(prediction) == 2:

        probabilities = prediction

        if (
            np.any(
                probabilities < 0
            )
            or
            np.any(
                probabilities > 1
            )
            or
            not np.isclose(
                np.sum(probabilities),
                1.0,
                atol=1e-3
            )
        ):

            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()

        class_index = int(
            np.argmax(
                probabilities
            )
        )

        if class_index == 0:

            label = "Normal"

        else:

            label = "Pneumonia"

        confidence = float(
            probabilities[class_index]
        )

        return (
            label,
            confidence,
            probabilities
        )

    # ========================================================
    # UNEXPECTED OUTPUT
    # ========================================================

    raise ValueError(
        f"""
Unexpected pneumonia model output.

Output length:
{len(prediction)}

Expected:
1 output for sigmoid OR
2 outputs for softmax.
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
    diagnosis_confidence=None,
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Pneumonia Detection System from X-ray Images",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Medical Image Analysis Report",
            ParagraphStyle(
                "SubTitle",
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=11,
                spaceAfter=15,
            )
        )
    )

    # --------------------------------------------------------
    # DATE/TIME
    # --------------------------------------------------------

    report_time = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Image Information",
            heading_style
        )
    )

    information_data = [
        [
            "Report Date",
            report_time
        ],
        [
            "Image Mode",
            "Grayscale"
        ],
        [
            "Detected Modality",
            modality
        ],
        [
            "Modality Confidence",
            f"{modality_confidence * 100:.2f}%"
        ],
    ]

    if diagnosis is not None:

        information_data.append(
            [
                "Pneumonia Result",
                diagnosis
            ]
        )

        information_data.append(
            [
                "Diagnosis Confidence",
                f"{diagnosis_confidence * 100:.2f}%"
            ]
        )

    table = Table(
        information_data,
        colWidths=[
            55 * mm,
            105 * mm
        ],
    )

    table.setStyle(
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
                ),
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    # --------------------------------------------------------
    # UPLOADED IMAGE
    # --------------------------------------------------------

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

    image_buffer.seek(
        0
    )

    report_image = RLImage(
        image_buffer,
        width=60 * mm,
        height=60 * mm,
    )

    story.append(
        report_image
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if diagnosis is not None:

        story.append(
            Paragraph(
                "Final Result",
                heading_style
            )
        )

        result_text = (
            f"<b>{diagnosis}</b>"
        )

        story.append(
            Paragraph(
                result_text,
                ParagraphStyle(
                    "Result",
                    parent=normal_style,
                    fontSize=15,
                    leading=20,
                    alignment=TA_CENTER,
                    spaceAfter=10,
                )
            )
        )

        story.append(
            Spacer(
                1,
                10
            )
        )

        story.append(
            Paragraph(
                "<b>Important:</b> This system is a research/decision-support "
                "application and is not a substitute for examination, "
                "diagnosis, or treatment by a qualified medical professional.",
                normal_style
            )
        )

    else:

        story.append(
            Paragraph(
                "This image was identified as an X-ray and is eligible "
                "for pneumonia analysis.",
                normal_style
            )
        )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    document.build(
        story
    )

    buffer.seek(
        0
    )

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Workflow"
    )

    st.markdown(
        """
        1. Upload an image
        2. Check image type
        3. Reject color images
        4. Accept grayscale image
        5. Detect modality
        6. X-ray → pneumonia model
        7. CT/MRI → reject for pneumonia analysis
        8. Generate PDF report
        """
    )

    st.divider()

    st.markdown(
        "**Required models:**"
    )

    st.code(
        """
modality_classifier.keras
best_xception_pneumonia_model.keras
        """,
        language="text"
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ],
    help=(
        "Upload a medical image. "
        "Color images will be rejected. "
        "Grayscale images will be analyzed."
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        )

        image.load()

    except Exception as e:

        st.error(
            f"Could not read the uploaded image: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="Input image",
        width="stretch"
    )

    # --------------------------------------------------------
    # Basic image information
    # --------------------------------------------------------

    st.write(
        f"**Image mode:** `{image.mode}`"
    )

    st.write(
        f"**Image size:** "
        f"`{image.width} × {image.height}` pixels"
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "Pneumonia Detection system form X ray Images",
        type="primary",
        width="stretch"
    )

    if analyze_button:

        # ====================================================
        # STEP 1 — COLOR IMAGE CHECK
        # ====================================================

        if not is_grayscale_image(
            image
        ):

            st.error(
                "Rejected: The uploaded image is a color image."
            )

            st.warning(
                "This system accepts grayscale medical images only."
            )

            st.stop()

        # ====================================================
        # GRAYSCALE IMAGE ACCEPTED
        # ====================================================

        st.success(
            "Image accepted: grayscale image detected."
        )

        # ====================================================
        # STEP 2 — LOAD MODALITY MODEL
        # ====================================================

        with st.spinner(
            "Identifying X-ray / CT / MRI..."
        ):

            try:

                modality_model = (
                    load_modality_model()
                )

            except Exception as e:

                st.error(
                    "Modality classification model could not be loaded."
                )

                st.code(
                    str(e)
                )

                st.stop()

        # ====================================================
        # STEP 3 — MODALITY PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing medical image modality..."
        ):

            try:

                (
                    modality,
                    modality_confidence,
                    modality_scores
                ) = predict_modality(
                    modality_model,
                    image
                )

            except Exception as e:

                st.error(
                    "Modality prediction failed."
                )

                st.code(
                    str(e)
                )

                st.stop()

        # ====================================================
        # SHOW MODALITY
        # ====================================================

        st.markdown(
            '<div class="result-box modality-result">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Detected Medical Image Type"
        )

        st.write(
            f"### {modality}"
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
        #
        # IMPORTANT:
        #
        # 0 = CT
        # 1 = MRI
        # 2 = X-ray
        #
        # Therefore:
        #
        # modality_scores[0] -> CT
        # modality_scores[1] -> MRI
        # modality_scores[2] -> X-ray
        #
        # ====================================================

        with st.expander(
            "View modality probabilities"
        ):

            # ------------------------------------------------
            # CT = index 0
            # ------------------------------------------------

            ct_probability = float(
                modality_scores[0]
            )

            st.write(
                f"CT: {ct_probability * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        ct_probability,
                        0.0
                    ),
                    1.0
                )
            )

            # ------------------------------------------------
            # MRI = index 1
            # ------------------------------------------------

            mri_probability = float(
                modality_scores[1]
            )

            st.write(
                f"MRI: {mri_probability * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        mri_probability,
                        0.0
                    ),
                    1.0
                )
            )

            # ------------------------------------------------
            # X-RAY = index 2
            # ------------------------------------------------

            xray_probability = float(
                modality_scores[2]
            )

            st.write(
                f"X-ray: {xray_probability * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        xray_probability,
                        0.0
                    ),
                    1.0
                )
            )

        # ====================================================
        # CT
        # ====================================================

        if modality == "CT":

            st.warning(
                "The uploaded image has been identified as a CT image."
            )

            st.info(
                "Pneumonia analysis using the proposed X-ray model "
                "has not been performed because this model is designed "
                "for chest X-ray images."
            )

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence,
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="medical_image_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.stop()

        # ====================================================
        # MRI
        # ====================================================

        if modality == "MRI":

            st.warning(
                "The uploaded image has been identified as an MRI image."
            )

            st.info(
                "Pneumonia analysis using the proposed X-ray model "
                "has not been performed because this model is designed "
                "for chest X-ray images."
            )

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence,
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="medical_image_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.stop()

        # ====================================================
        # X-RAY
        # ====================================================

        if modality == "X-ray":

            st.success(
                "X-ray detected. Proceeding to pneumonia analysis."
            )

            # ------------------------------------------------
            # LOAD PROPOSED MODEL
            # ------------------------------------------------

            with st.spinner(
                "Loading proposed pneumonia detection model..."
            ):

                try:

                    pneumonia_model = (
                        load_pneumonia_model()
                    )

                except Exception as e:

                    st.error(
                        "The proposed pneumonia model could not be loaded."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()

            # ------------------------------------------------
            # PNEUMONIA PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "Analyzing X-ray for pneumonia..."
            ):

                try:

                    (
                        diagnosis,
                        diagnosis_confidence,
                        raw_prediction
                    ) = predict_pneumonia(
                        pneumonia_model,
                        image
                    )

                except Exception as e:

                    st.error(
                        "Pneumonia prediction failed."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()

            # =================================================
            # SHOW FINAL RESULT
            # =================================================

            if diagnosis == "Pneumonia":

                st.markdown(
                    '<div class="result-box pneumonia-result">',
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
                    '<div class="result-box normal-result">',
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
            # PDF REPORT
            # =================================================

            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence,
                diagnosis=diagnosis,
                diagnosis_confidence=diagnosis_confidence,
            )

            st.download_button(
                label="Download Final Report (PDF)",
                data=pdf_bytes,
                file_name="pneumonia_detection_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
