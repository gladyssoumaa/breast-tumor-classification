from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mammogram_cnn_best.keras"

IMAGE_SIZE = (224, 224)




CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}

UNCERTAINTY_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.70


st.set_page_config(
    page_title="Mammographic Mass Classification",
    page_icon="🩺",
    layout="wide"
)


st.title("Mammographic Mass Classification Portal")

st.markdown(
    """
    The model is a convolutional neural network
    (CNN) used to classify mammographic images into two categories:

    - **Benign**
    - **Malignant**

    The system is designed for mammography images of the breast.
    """
)

st.warning(
    """
    This application is for academic and research demonstration only.
    It must not be used to diagnose breast cancer, make treatment
    decisions, or replace assessment by a qualified healthcare professional.
    """
)


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )



def preprocess_image(uploaded_file):
    
    image_bytes = uploaded_file.getvalue()

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


def predict_image(model, uploaded_file):

    processed_image = preprocess_image(
        uploaded_file
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    malignant_probability = float(
        np.squeeze(prediction)
    )

    malignant_probability = np.clip(
        malignant_probability,
        0.0,
        1.0
    )

    benign_probability = (
        1.0 - malignant_probability
    )

    if malignant_probability >= 0.5:

        predicted_class = 1
        confidence = malignant_probability

    else:

        predicted_class = 0
        confidence = benign_probability

    return (
        predicted_class,
        confidence,
        benign_probability,
        malignant_probability
    )


if not MODEL_PATH.exists():

    st.error(
        f"""
        The trained model could not be found.

        Expected model:

        `{MODEL_PATH.name}`

        """
    )

    st.stop()


try:

    model = load_model(
        MODEL_PATH
    )

except Exception as error:

    st.error(
        "The model could not be loaded."
    )

    st.exception(error)

    st.stop()


with st.expander("Model Information"):

    st.write(
        "**Model:** Convolutional Neural Network"
    )

    st.write(
        f"**Model file:** `{MODEL_PATH.name}`"
    )

    st.write(
        f"**Expected input shape:** `{model.input_shape}`"
    )

    st.write(
        f"**Output shape:** `{model.output_shape}`"
    )

    st.write(
        "**Input:** Mammography JPEG image"
    )

    st.write(
        "**Preprocessing:** Grayscale → 224 × 224 → "
        "float32 → pixel values divided by 255"
    )

    st.write(
        "**Output:** Probability of malignancy"
    )


st.header("Upload Mammography Image")

st.write(
    """
    Upload a JPG or JPEG mammography image. The image will be
    automatically converted to grayscale, resized to 224 × 224 pixels,
    normalized, and passed to the trained CNN.
    """
)


uploaded_file = st.file_uploader(
    "Choose a mammography image",
    type=["jpg", "jpeg"]
)


if uploaded_file is not None:

    try:

        preview_image = Image.open(
            uploaded_file
        )

        preview_image.load()

    except Exception:

        st.error(
            "The uploaded file could not be read as an image."
        )

        st.stop()


    st.markdown("---")

    image_column, result_column = st.columns(
        [1, 1]
    )


    with image_column:

        st.subheader("Uploaded Image")

        st.image(
            preview_image,
            caption=uploaded_file.name,
            use_container_width=True
        )

        st.caption(
            f"Original image size: "
            f"{preview_image.width} × "
            f"{preview_image.height} pixels"
        )


    with result_column:

        st.subheader("Model Prediction")

        with st.spinner(
            "Analysing the mammography image..."
        ):

            (
                predicted_class,
                confidence,
                benign_probability,
                malignant_probability
            ) = predict_image(
                model,
                uploaded_file
            )


        predicted_label = CLASS_NAMES[
            predicted_class
        ]

        confidence_percentage = (
            confidence * 100
        )


        if confidence < UNCERTAINTY_THRESHOLD:

            st.warning(
                "The model is uncertain about this image."
            )

            st.metric(
                "Prediction confidence",
                f"{confidence_percentage:.1f}%"
            )

            st.write(
                """
                The model does not have strong confidence in either
                class. This result should not be interpreted as a
                reliable classification.
                """
            )


        elif predicted_class == 1:

            if confidence >= CONFIDENCE_THRESHOLD:

                st.error(
                    f"Prediction: {predicted_label}"
                )

            else:

                st.warning(
                    f"Prediction: {predicted_label}"
                )

            st.metric(
                "Confidence",
                f"{confidence_percentage:.1f}%"
            )

            st.info(
                f"""
                The model predicts that this mammographic image
                belongs to the **malignant** class with an estimated
                confidence of **{confidence_percentage:.1f}%**.
                """
            )


        else:

            if confidence >= CONFIDENCE_THRESHOLD:

                st.success(
                    f"Prediction: {predicted_label}"
                )

            else:

                st.warning(
                    f"Prediction: {predicted_label}"
                )

            st.metric(
                "Confidence",
                f"{confidence_percentage:.1f}%"
            )

            st.info(
                f"""
                The model predicts that this mammographic image
                belongs to the **benign** class with an estimated
                confidence of **{confidence_percentage:.1f}%**.
                """ 
            )


    st.markdown("---")

    st.subheader("Prediction Probabilities")


    probability_column_1, probability_column_2 = st.columns(
        2
    )


    with probability_column_1:

        st.metric(
            "Benign probability",
            f"{benign_probability * 100:.1f}%"
        )


    with probability_column_2:

        st.metric(
            "Malignant probability",
            f"{malignant_probability * 100:.1f}%"
        )


    st.progress(
        malignant_probability,
        text=(
            f"Malignant probability: "
            f"{malignant_probability * 100:.1f}%"
        )
    )


    st.markdown("---")

    st.subheader("How to Interpret This Result")

    st.write(
        """
        The probabilities represent the CNN's estimated
        probabilities for the benign and malignant classes.

        A high model confidence does not mean that the prediction is
        medically certain. The model has been developed for academic
        coursework and has not been clinically validated.
        """
    )


st.markdown("---")


with st.expander("About This Project"):

    st.markdown(
        """
        ### Mammographic Mass Classification

        This project investigates the use of a convolutional neural
        network to classify mammography images into two classes:

        - **Benign:** associated with non-cancerous pathology.
        - **Malignant:** associated with cancerous pathology.

        The CNN receives a grayscale mammography image and produces
        a probability representing how strongly the model predicts
        the malignant class.

        ### Model preprocessing

        Uploaded JPEG images are processed using the same basic
        preprocessing pipeline used during model development:

        1. Decode the JPEG image.
        2. Convert it to one grayscale channel.
        3. Resize it to 224 × 224 pixels.
        4. Convert pixel values to float32.
        5. Divide pixel values by 255.
        6. Pass the processed image to the CNN.

        ### Important limitation

        This application is a student coursework prototype.

        It is not a medical device and must not be used to diagnose
        breast cancer or make clinical decisions.

        Clinical diagnosis requires assessment by qualified healthcare
        professionals and may require additional imaging, examination,
        biopsy, or other diagnostic procedures.
        """
    )