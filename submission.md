# Submission Note 

**Student:** Gladys Ouma

**Project:** CNN-Based Classification of Mammographic Masses — Benign vs Malignant 

**Date:** 15 September 2026

## Local run verification

I verified that the application runs locally with the required dependencies installed:

- Python version: 3.10.2
- TensorFlow version: 2.21.0
- Steps performed:
  1. `pip install -r requirements.txt`
  2. `streamlit run streamlit_app.py`
  3. The trained CNN model `mammogram_cnn_best.keras`, exported from the notebook, loads successfully through `@st.cache_resource`.
  4. Uploaded mammography images are preprocessed using the same procedure as the training pipeline: grayscale conversion, resizing to 224 × 224 pixels, and normalization to the [0, 1] range.
  5. Uploaded test images produce a real, specific prediction consisting of the predicted class and confidence/probabilities.
  6. App behaviour was checked using an image that the CNN classifies incorrectly. The incorrect prediction is displayed honestly and is consistent with the model's failure analysis reported in A6.

## Files submitted

| File | Purpose |
| --- | --- |
| `Medical Imaging ML Project.ipynb` | Part A: full documentation, preprocessing, CNN training, evaluation, and analysis (A1–A8) |
| `streamlit_app.py` | Part B: Streamlit diagnostic portal |
| `requirements.txt` | Required Python package versions |
| `README.md` | Setup and application run instructions |
| `mammogram_cnn_best.keras` | Trained CNN model exported from the notebook |
| `Submission Note.md` | This verification statement |