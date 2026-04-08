import streamlit as st
import pickle
import os

# Correct pdfminer import
from pdfminer.high_level import extract_text
import docx2txt

from utils.text_cleaning import clean_text
from utils.resume_parser import extract_email, extract_phone, extract_skills

# -----------------------------
# Load trained models safely
# -----------------------------
PIPELINE_PATH = os.path.join("models", "pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

if not os.path.exists(PIPELINE_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
    st.error("Model files not found. Please train the model first.")
    st.stop()

pipeline = pickle.load(open(PIPELINE_PATH, "rb"))
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))

# -----------------------------
# Skill list (expand anytime)
# -----------------------------
SKILL_LIST = [
    "Python", "Java", "SQL", "Machine Learning", "Data Science",
    "Deep Learning", "NLP", "Pandas", "NumPy", "Excel",
    "Power BI", "Tableau", "AWS", "Docker", "Git"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Parser", layout="centered")

st.title("📄 Resume Parser & Job Role Predictor")
st.write("Upload your resume (TXT, PDF, or DOCX)")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["txt", "pdf", "docx"]
)

# -----------------------------
# File Processing
# -----------------------------
if uploaded_file is not None:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    try:
        # -------- TEXT EXTRACTION --------
        if file_extension == "txt":
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_extension == "pdf":
            # Save temporarily (more stable for pdfminer)
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_file.read())
            text = extract_text("temp_resume.pdf")
            os.remove("temp_resume.pdf")

        elif file_extension == "docx":
            text = docx2txt.process(uploaded_file)

        else:
            st.error("Unsupported file format")
            st.stop()

        if not text or len(text.strip()) == 0:
            st.error("Could not extract text from the resume.")
            st.stop()

        # -------- CLEAN TEXT --------
        cleaned_text = clean_text(text)

        # -------- RESUME PARSING --------
        email = extract_email(text)
        phone = extract_phone(text)
        skills_found = extract_skills(text, SKILL_LIST)

        # -------- JOB ROLE PREDICTION --------
        prediction = pipeline.predict([cleaned_text])
        job_role = label_encoder.inverse_transform(prediction)[0]

        # -------- CONFIDENCE SCORE (NEW ADDITION) --------
        try:
            proba = pipeline.predict_proba([cleaned_text])
            confidence = max(proba[0]) * 100
        except:
            confidence = None

        # -----------------------------
        # Display Results
        # -----------------------------
        st.success("✅ Resume processed successfully!")

        st.subheader("📌 Parsed Resume Details")
        st.write(f"**Email:** {email if email else 'Not found'}")
        st.write(f"**Phone:** {phone if phone else 'Not found'}")
        st.write(f"**Skills:** {', '.join(skills_found) if skills_found else 'Not found'}")

        st.subheader("💼 Predicted Job Role")
        st.write(f"### {job_role}")

        if confidence:
            st.write(f"**Confidence Score:** {confidence:.2f}%")

    except Exception as e:
        st.error("An error occurred while processing the resume.")
        st.exception(e)