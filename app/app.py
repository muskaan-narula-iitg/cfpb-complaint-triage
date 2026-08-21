
"""
Streamlit demo for the CFPB complaint triage system.
"""
import sys
from pathlib import Path
 
import joblib
import numpy as np
import streamlit as st
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from resolution_suggester import suggest_resolution
from ood_detector import OODDetector
 
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
 
st.set_page_config(page_title="Complaint Triage Demo", page_icon="🏦")
st.title("🏦 Financial Complaint Triage")
st.caption("TF-IDF + Logistic Regression baseline, trained on real CFPB complaints.")
 
 
@st.cache_resource
def load_models():
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    category_clf = joblib.load(MODEL_DIR / "tfidf_category_logreg.joblib")
    ood_detector = OODDetector.load(MODEL_DIR / "ood_detector.pkl")
    return vectorizer, category_clf, ood_detector
 
 
def predict_category(text, vectorizer, clf):
    X = vectorizer.transform([text])
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        confidence = probs.max()
    else:
        margins = clf.decision_function(X)[0]
        confidence = float(1 / (1 + np.exp(-np.max(margins))))
    pred = clf.predict(X)[0]
    return pred, confidence
 
 
def triage(text, vectorizer, category_clf, ood_detector):
    category, confidence = predict_category(text, vectorizer, category_clf)
    is_ood, reason, details = ood_detector.is_ood(text, confidence)
 
    result = {"query": text, "in_scope": not is_ood}
    if is_ood:
        result["reason"] = reason
        return result
 
    result["category"] = category
    result["confidence"] = confidence
    result["resolution"] = suggest_resolution(category)
    return result
 
 
try:
    vectorizer, category_clf, ood_detector = load_models()
except FileNotFoundError:
    st.error("Models not found. Run notebooks 01, 02, and 04 first to train and save them.")
    st.stop()
 
query = st.text_area(
    "Describe your issue:",
    placeholder="e.g. There's a charge on my credit card statement I don't recognize...",
)
 
if st.button("Triage this complaint") and query.strip():
    result = triage(query, vectorizer, category_clf, ood_detector)
    if result["in_scope"]:
        st.success(f"**Category:** {result['category']}  (confidence: {result['confidence']:.1%})")
        st.info(f"**Suggested next step:** {result['resolution']}")
    else:
        st.warning(f"This doesn't look like a financial complaint — {result['reason']}")