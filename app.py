# 1. Imports
import os
import logging
import copy
import pandas as pd
import numpy as np
from io import BytesIO
import statistics  # Import the whole module for StatisticsError
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from flask import Flask, render_template, request, send_file, session, redirect, url_for

# 2. Initialize Flask App
app = Flask(__name__)

# 3. Configure App
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_default_development_secret_key_123!")

# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# 4. Define Constants
DATA_PATH = "Training.csv"
TEST_DATA_PATH = "Testing.csv"
STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')
IMAGE_PATH = os.path.join(STATIC_PATH, 'images')
REPORTS_PATH = os.path.join(STATIC_PATH, 'reports') # Define reports path

# Ensure necessary directories exist
for path in [STATIC_PATH, IMAGE_PATH, REPORTS_PATH]:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        except OSError as e:
            logging.error(f"Failed to create directory {path}: {e}")

# --- START: DISEASE_PRECAUTIONS ---
# (Keep DISEASE_PRECAUTIONS dictionary here)
DISEASE_PRECAUTIONS = {
    "Fungal infection": ["Maintain good hygiene; keep skin dry and clean.", "Wear breathable clothing (cotton).", "Use antifungal creams/powders as prescribed."],
    "Allergy": ["Identify/avoid known allergens.", "Keep your environment clean and dust-free.", "Take antihistamines as prescribed."],
    "GERD": ["Avoid trigger foods (spicy, fatty, acidic, etc.).", "Eat smaller, more frequent meals.", "Avoid lying down 2-3 hours after eating."],
    "Chronic cholestasis": ["Follow your doctor's dietary recommendations.", "Avoid alcohol/liver-damaging substances.", "Take medications as prescribed."],
    "Drug Reaction": ["Stop the suspected medication, contact your doctor.", "Take antihistamines for itching/rash.", "Use topical corticosteroids as prescribed."],
    "Peptic ulcer diseae": ["Avoid NSAIDs (ibuprofen, aspirin).", "Avoid smoking and alcohol.", "Take medications as prescribed (PPIs, antibiotics)."],
    "AIDS": ["Adhere to antiretroviral therapy (ART).", "Practice safe sex to prevent transmission.", "Maintain a healthy lifestyle."],
    "Diabetes": ["Follow a healthy diet plan (low sugar, processed foods).", "Monitor blood sugar levels regularly.", "Take prescribed medications as directed."],
    "Gastroenteritis": ["Drink plenty of fluids to prevent dehydration.", "Eat bland, easily digestible foods.", "Get plenty of rest."],
    "Bronchial Asthma": ["Avoid known asthma triggers.", "Use prescribed inhalers as directed.", "Develop an asthma action plan with your doctor."],
    "Hypertension": ["Follow a healthy diet (low sodium/fat).", "Engage in regular physical activity.", "Monitor blood pressure regularly."],
    "Migraine": ["Identify/avoid migraine triggers.", "Maintain a regular sleep schedule.", "Take pain relievers as prescribed."],
    "Cervical spondylosis": ["Maintain good posture.", "Perform gentle neck exercises.", "Apply heat/ice to the neck for pain relief."],
    "Paralysis (brain hemorrhage)": ["Follow your doctor's rehabilitation plan.", "Engage in physical therapy.", "Address emotional challenges with counseling."],
    "Jaundice": ["Treat the underlying cause of jaundice.", "Avoid alcohol and liver-damaging substances.", "Follow your doctor's dietary recommendations."],
    "hepatitis A": ["Practice good hygiene (handwashing).", "Avoid contaminated food/water.", "Get vaccinated against hepatitis A."],
    "Hepatitis B": ["Get vaccinated against hepatitis B.", "Practice safe sex to prevent transmission.", "Avoid sharing needles."],
    "Hepatitis C": ["Avoid sharing needles.", "Practice safe sex to prevent transmission.", "Take antiviral medications as prescribed."],
    "Hepatitis D": ["Get vaccinated against hepatitis B.", "Avoid sharing needles.", "Practice safe sex to prevent transmission."],
    "Hepatitis E": ["Practice good hygiene (handwashing).", "Avoid contaminated food/water.", "Rest and stay hydrated."],
    "Alcoholic hepatitis": ["Stop drinking alcohol completely.", "Follow your doctor's dietary recommendations.", "Seek support/counseling."],
    "Tuberculosis": ["Take all medications exactly as directed.", "Cover mouth/nose when coughing/sneezing.", "Ensure good ventilation."],
    "Common Cold": ["Rest and get plenty of sleep.", "Drink plenty of fluids.", "Use a humidifier."],
    "Pneumonia": ["Take antibiotics/antivirals as prescribed.", "Get plenty of rest.", "Drink plenty of fluids."],
    "Dimorphic hemmorhoids(piles)": ["Increase fiber intake.", "Drink plenty of fluids.", "Sitz baths (warm water soak)."],
    "Heart attack": ["Call emergency services immediately.", "Take aspirin as directed.", "Follow doctor's recommendations for lifestyle."],
    "Varicose veins": ["Elevate your legs when sitting/lying.", "Wear compression stockings.", "Avoid standing/sitting long periods."],
    "Hypothyroidism": ["Take thyroid hormone replacement medication.", "Get regular blood tests.", "Follow your doctor's dietary recommendations."],
    "Hyperthyroidism": ["Take antithyroid medications.", "Radioactive iodine therapy (may be used).", "Follow doctor's dietary recommendations."],
    "Hypoglycemia": ["Eat regular meals/snacks.", "Carry fast-acting glucose.", "Monitor blood sugar levels."],
    "Osteoarthristis": ["Engage in regular exercise.", "Maintain a healthy weight.", "Apply heat/ice to affected joints."],
    "Arthritis": ["Engage in regular exercise.", "Maintain a healthy weight.", "Take medications as prescribed."],
    "(vertigo) Paroymsal  Positional Vertigo": ["Follow doctor's vestibular rehab exercises.", "Avoid sudden head movements.", "Use caution changing positions."],
    "Acne": ["Wash your face gently twice a day.", "Avoid scrubbing/picking at pimples.", "Apply topical acne medications as prescribed."],
    "Urinary tract infection": ["Drink plenty of fluids.", "Urinate frequently.", "Take antibiotics as prescribed."],
    "Psoriasis": ["Use topical medications as prescribed.", "Moisturize your skin regularly.", "Avoid triggers (stress, smoking)."],
    "Impetigo": ["Wash the affected area gently.", "Apply topical antibiotic ointment.", "Wash your hands frequently."],
    "Dengue": ["Rest and drink plenty of fluids", "Take acetaminophen or paracetamol for fever/pain.", "Avoid mosquito bites (repellent, long sleeves, screens)."],
    "Typhoid": ["Take antibiotics as prescribed.", "Drink only bottled or boiled water.", "Avoid raw fruits/vegetables you can't peel."],
    "Chicken pox": ["Rest and drink plenty of fluids", "Avoid scratching.", "Use calamine lotion or oatmeal baths for itching."],
    "Malaria": ["Take antimalarial medication as prescribed.", "Avoid mosquito bites (repellent, long sleeves, screens).", "Sleep under a mosquito net."]
}
# --- END: DISEASE_PRECAUTIONS ---

# --- START: DOCTOR_SPECIALISTS ---
# (Keep DOCTOR_SPECIALISTS dictionary here)
DOCTOR_SPECIALISTS = {
    "Fungal infection": ["Dermatologist"],
    "Allergy": ["Allergist", "Immunologist"],
    "GERD": ["Gastroenterologist"],
    "Chronic cholestasis": ["Gastroenterologist", "Hepatologist"],
    "Drug Reaction": ["Allergist", "Dermatologist"],
    "Peptic ulcer diseae": ["Gastroenterologist"],
    "AIDS": ["Infectious Disease Specialist", "Immunologist"],
    "Diabetes": ["Endocrinologist"],
    "Gastroenteritis": ["Gastroenterologist"],
    "Bronchial Asthma": ["Pulmonologist", "Allergist"],
    "Hypertension": ["Cardiologist", "Nephrologist"],
    "Migraine": ["Neurologist"],
    "Cervical spondylosis": ["Neurologist", "Orthopedist"],
    "Paralysis (brain hemorrhage)": ["Neurologist", "Rehabilitation Specialist"],
    "Jaundice": ["Gastroenterologist", "Hepatologist"],
    "hepatitis A": ["Gastroenterologist", "Hepatologist"],
    "Hepatitis B": ["Gastroenterologist", "Hepatologist"],
    "Hepatitis C": ["Gastroenterologist", "Hepatologist"],
    "Hepatitis D": ["Gastroenterologist", "Hepatologist"],
    "Hepatitis E": ["Gastroenterologist", "Hepatologist"],
    "Alcoholic hepatitis": ["Gastroenterologist", "Hepatologist"],
    "Tuberculosis": ["Pulmonologist", "Infectious Disease Specialist"],
    "Common Cold": ["General Practitioner"],
    "Pneumonia": ["Pulmonologist", "Infectious Disease Specialist"],
    "Dimorphic hemmorhoids(piles)": ["Proctologist", "General Surgeon"],
    "Heart attack": ["Cardiologist"],
    "Varicose veins": ["Vascular Surgeon"],
    "Hypothyroidism": ["Endocrinologist"],
    "Hyperthyroidism": ["Endocrinologist"],
    "Hypoglycemia": ["Endocrinologist"],
    "Osteoarthristis": ["Orthopedist", "Rheumatologist"],
    "Arthritis": ["Rheumatologist"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Neurologist", "ENT Specialist"],
    "Acne": ["Dermatologist"],
    "Urinary tract infection": ["Urologist", "Nephrologist"],
    "Psoriasis": ["Dermatologist"],
    "Impetigo": ["Dermatologist", "General Practitioner"],
    "Dengue": ["Infectious Disease Specialist", "General Practitioner"],
    "Typhoid": ["Infectious Disease Specialist", "General Practitioner"],
    "Chicken pox": ["General Practitioner", "Pediatrician"],
    "Malaria": ["Infectious Disease Specialist", "General Practitioner"]
}
# --- END: DOCTOR_SPECIALISTS ---

# 5. Define Helper Functions

def load_and_preprocess_data(data_path, test_data_path=None):
    # (Keep as is)
    """Loads and preprocesses the training and potentially testing data."""
    logging.info("Loading and preprocessing data...")
    try:
        data = pd.read_csv(data_path).dropna(axis=1, how='all'); data.dropna(subset=['prognosis'], inplace=True)
        encoder = LabelEncoder(); X_cols = [col for col in data.columns if col != 'prognosis']
        X = data[X_cols]; y = data["prognosis"]; y_encoded = encoder.fit_transform(y); X = X.fillna(0)
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        X_test, y_test_encoded = X_val, y_val_encoded
        if test_data_path and os.path.exists(test_data_path):
            logging.info(f"Loading testing data from {test_data_path}")
            try:
                test_data = pd.read_csv(test_data_path).dropna(axis=1, how='all'); test_data.dropna(subset=['prognosis'], inplace=True)
                missing_cols = set(X_train.columns) - set(test_data.columns); [setattr(test_data, c, 0) for c in missing_cols]
                extra_cols = set(test_data.columns) - set(X_train.columns) - {'prognosis'}; test_data = test_data.drop(columns=list(extra_cols))
                test_data = test_data[X_train.columns.tolist() + ['prognosis']]; X_test_data = test_data[X_train.columns]
                y_test_data = test_data['prognosis']; y_test_encoded = encoder.transform(y_test_data); X_test = X_test_data.fillna(0)
                logging.info("Using provided testing data.")
            except Exception as e: logging.error(f"Error processing test data file {test_data_path}: {e}. Using validation split.", exc_info=True); X_test, y_test_encoded = X_val, y_val_encoded
        else: logging.info("No separate testing data. Using validation split as test set.")
        logging.info("Data loaded and preprocessed successfully.")
        return X_train, X_test, y_train_encoded, y_test_encoded, encoder, X_cols
    except FileNotFoundError: logging.error(f"Training data file not found: {data_path}"); raise
    except Exception as e: logging.exception("Error during data loading/preprocessing:"); raise

def train_models(X_train, y_train):
    # (Keep as is)
    """Trains the machine learning models."""
    logging.info("Training models...")
    try:
        svm_model = SVC(probability=True, random_state=42); nb_model = GaussianNB(); knn_model = KNeighborsClassifier(n_neighbors=5)
        svm_model.fit(X_train, y_train); logging.info("SVM model trained.")
        nb_model.fit(X_train, y_train); logging.info("Naive Bayes model trained.")
        knn_model.fit(X_train, y_train); logging.info("KNN model trained.")
        logging.info("All models trained successfully.")
        return svm_model, nb_model, knn_model
    except Exception as e: logging.exception("Error during model training:"); raise

def create_symptom_index(symptoms_list):
    # (Keep as is)
    """Creates a symptom index dictionary {formatted_symptom_name: column_index}."""
    logging.info("Creating symptom index...")
    try:
        symptom_index = {" ".join([word.capitalize() for word in s.replace('_', ' ').split()]): idx for idx, s in enumerate(symptoms_list)}
        logging.info(f"Symptom index created successfully with {len(symptom_index)} entries.")
        return symptom_index
    except Exception as e: logging.exception("Error creating symptom index:"); return {}

def predict_disease(symptoms_list_input, symptom_index, encoder, svm_model, nb_model, knn_model, all_symptom_columns, X_train_for_acc, y_train_for_acc):
    # (Keep as is)
    """Predicts disease based on input symptoms, calculates accuracy on training data. (Enhanced Debugging)"""
    logging.info(f"Predicting disease for symptoms: {symptoms_list_input}")
    if not symptoms_list_input: return {"final_prediction": "No symptoms provided", "precautions": [], "specialists": [], "total_accuracy": 0.0, "svm_model_prediction": "N/A", "naive_bayes_prediction": "N/A", "knn_model_prediction": "N/A"}
    try:
        input_data_dict = {col: 0 for col in all_symptom_columns}; symptoms_found_count = 0; matched_symptoms = []
        for symptom in symptoms_list_input:
            standardized_symptom = " ".join([word.capitalize() for word in symptom.replace('_', ' ').strip().split()])
            if standardized_symptom in symptom_index:
                col_index = symptom_index[standardized_symptom]
                if 0 <= col_index < len(all_symptom_columns):
                    original_col_name = all_symptom_columns[col_index]; input_data_dict[original_col_name] = 1
                    symptoms_found_count += 1; matched_symptoms.append(standardized_symptom)
                    logging.debug(f"Matched symptom: '{symptom}' -> '{standardized_symptom}' -> Col: '{original_col_name}'")
                else: logging.warning(f"Symptom '{symptom}' index {col_index} out of bounds.")
            else: logging.warning(f"Symptom '{symptom}' (std: '{standardized_symptom}') not found.")
        if symptoms_found_count == 0: return {"final_prediction": "No valid symptoms matched", "precautions": [], "specialists": [], "total_accuracy": 0.0, "svm_model_prediction": "N/A", "naive_bayes_prediction": "N/A", "knn_model_prediction": "N/A"}
        logging.info(f"Matched {symptoms_found_count} symptoms: {matched_symptoms}")
        input_df = pd.DataFrame([input_data_dict], columns=all_symptom_columns)
        logging.debug(f"Input DataFrame for prediction:\n{input_df.to_string()}")
        svm_pred_encoded = svm_model.predict(input_df)[0]; nb_pred_encoded = nb_model.predict(input_df)[0]; knn_pred_encoded = knn_model.predict(input_df)[0]
        logging.info(f"Encoded Predictions: SVM={svm_pred_encoded}, NB={nb_pred_encoded}, KNN={knn_pred_encoded}")
        svm_prediction = "Decoding Error"; nb_prediction = "Decoding Error"; knn_prediction = "Decoding Error"
        try: svm_prediction = encoder.inverse_transform([svm_pred_encoded])[0]; logging.info(f"**** Decoded SVM Prediction: {svm_prediction}")
        except ValueError: logging.error(f"SVM label {svm_pred_encoded} not in encoder.", exc_info=True)
        except Exception as e: logging.error(f"Error decoding SVM: {e}", exc_info=True)
        try: nb_prediction = encoder.inverse_transform([nb_pred_encoded])[0]; logging.info(f"**** Decoded NB Prediction: {nb_prediction}")
        except ValueError: logging.error(f"NB label {nb_pred_encoded} not in encoder.", exc_info=True)
        except Exception as e: logging.error(f"Error decoding NB: {e}", exc_info=True)
        try: knn_prediction = encoder.inverse_transform([knn_pred_encoded])[0]; logging.info(f"**** Decoded KNN Prediction: {knn_prediction}")
        except ValueError: logging.error(f"KNN label {knn_pred_encoded} not in encoder.", exc_info=True)
        except Exception as e: logging.error(f"Error decoding KNN: {e}", exc_info=True)
        valid_predictions = [p for p in [svm_prediction, nb_prediction, knn_prediction] if p != "Decoding Error"]
        logging.info(f"Valid decoded predictions for mode: {valid_predictions}")
        if not valid_predictions: final_prediction = "Prediction Failed (Decoding)"; logging.error("All models failed decoding.")
        else:
            try: final_prediction = mode(valid_predictions)
            except statistics.StatisticsError: logging.warning(f"No mode in {valid_predictions}. Fallback."); final_prediction = valid_predictions[0]
        total_accuracy = 0.0
        try:
             svm_accuracy = accuracy_score(y_train_for_acc, svm_model.predict(X_train_for_acc)); nb_accuracy = accuracy_score(y_train_for_acc, nb_model.predict(X_train_for_acc)); knn_accuracy = accuracy_score(y_train_for_acc, knn_model.predict(X_train_for_acc))
             total_accuracy = (svm_accuracy + nb_accuracy + knn_accuracy) / 3
             logging.info(f"Acc (Train): SVM={svm_accuracy:.4f}, NB={nb_accuracy:.4f}, KNN={knn_accuracy:.4f}, Avg={total_accuracy:.4f}")
        except Exception as acc_e: logging.error(f"Accuracy calc error: {acc_e}")
        precautions = DISEASE_PRECAUTIONS.get(final_prediction, ["Consult healthcare professional."])
        specialists = DOCTOR_SPECIALISTS.get(final_prediction, ["General Practitioner"])
        predictions_result = {"svm_model_prediction": svm_prediction, "naive_bayes_prediction": nb_prediction, "knn_model_prediction": knn_prediction, "final_prediction": final_prediction, "precautions": precautions, "specialists": specialists, "total_accuracy": total_accuracy}
        logging.info(f"**** FINAL predictions_result dictionary: {predictions_result}")
        return predictions_result
    except Exception as e:
        logging.exception("Unexpected error during predict_disease:")
        return {"svm_model_prediction": "Error", "naive_bayes_prediction": "Error", "knn_model_prediction": "Error", "final_prediction": "Prediction Error", "precautions": ["Error occurred."], "specialists": ["Error"], "total_accuracy": 0.0}

# --- PDF FUNCTION with ALL Syntax Corrections ---
def create_pdf_report(filepath, patient_details, predictions, hospital_details, hospital_logo_path):
    """Creates PDF report with corrected layout and syntax."""
    logging.info(f"Creating PDF report at: {filepath}")
    try:
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter

        # ========== Styles ==========
        styles = getSampleStyleSheet()
        title_style = copy.deepcopy(styles["Title"])
        title_style.alignment = TA_CENTER
        title_style.fontSize = 18
        title_style.leading = 22
        title_style.textColor = '#2c3e50'
        title_style.fontName = 'Helvetica-Bold'

        heading_style = copy.deepcopy(styles["Heading2"])
        heading_style.fontSize = 11
        heading_style.textColor = '#ffffff'
        heading_style.alignment = TA_CENTER
        heading_style.fontName = 'Helvetica-Bold'
        heading_style.leftPadding = 6
        heading_style.rightPadding = 6

        body_style = copy.deepcopy(styles["BodyText"])
        body_style.fontSize = 10
        body_style.leading = 14
        body_style.textColor = '#34495e'
        body_style.alignment = TA_LEFT

        label_style = copy.deepcopy(styles["BodyText"])
        label_style.fontName = 'Helvetica-Bold'
        label_style.fontSize = 9
        label_style.textColor = '#2c3e50'
        label_style.alignment = TA_LEFT

        value_style = copy.deepcopy(styles["BodyText"])
        value_style.fontSize = 9
        value_style.textColor = '#7f8c8d'
        value_style.alignment = TA_LEFT

        # ========== Header Section ==========
        y_position = height - 0.5 * inch
        left_margin = 0.5 * inch
        right_margin = width - 0.5 * inch

        if hospital_logo_path and os.path.exists(hospital_logo_path):
            try:
                c.drawImage(hospital_logo_path, left_margin, y_position - 0.7 * inch, width=1.0 * inch, height=0.7 * inch, preserveAspectRatio=True, anchor='n')
                logging.info(f"Drawing logo: {hospital_logo_path}")
            except Exception as img_err:
                logging.error(f"Could not draw logo {hospital_logo_path}: {img_err}")
        elif hospital_logo_path:
            logging.warning(f"Logo file not found: {hospital_logo_path}")

        c.setFont("Helvetica-Bold", 9)
        c.setFillColor('#2c3e50')
        c.drawRightString(right_margin, y_position - 0.2 * inch, hospital_details.get('name', 'Healthcare Center'))

        c.setFont("Helvetica", 8)
        c.setFillColor('#7f8c8d')
        c.drawRightString(right_margin, y_position - 0.35 * inch, hospital_details.get('address', '123 Health St'))
        c.drawRightString(right_margin, y_position - 0.5 * inch, f"Tel: {hospital_details.get('phone', 'N/A')}")
        c.drawRightString(right_margin, y_position - 0.65 * inch, f"Email: {hospital_details.get('email', 'N/A')}")
        y_position -= 1.0 * inch

        # Report Title
        title_para = Paragraph("MEDICAL DIAGNOSIS REPORT", title_style)
        title_w, title_h = title_para.wrapOn(c, width - 2 * left_margin, 1 * inch)
        title_para.drawOn(c, left_margin, y_position - title_h)
        y_position -= (title_h + 0.3 * inch)
        c.setStrokeColor('#bdc3c7')
        c.setLineWidth(0.5)
        c.line(left_margin, y_position, right_margin, y_position)
        y_position -= 0.3 * inch

        # ========== Patient Information Section ==========
        section_padding = 0.1 * inch
        section_header_height = 0.25 * inch
        c.setFillColor('#3498db')
        c.setStrokeColor('#3498db')
        c.roundRect(left_margin, y_position - section_header_height, right_margin - left_margin, section_header_height, radius=3, fill=1, stroke=1)

        p_info_header = Paragraph("PATIENT INFORMATION", heading_style)
        p_w, p_h = p_info_header.wrapOn(c, right_margin - left_margin - 2 * heading_style.leftPadding, section_header_height)
        p_info_header.drawOn(c, left_margin + heading_style.leftPadding, y_position - p_h - (section_header_height - p_h)/2)
        y_position -= (section_header_height + 0.3 * inch)

        details = [
            ("Full Name:", f"{patient_details.get('title','')} {patient_details.get('first_name','')} {patient_details.get('last_name','')}"),
            ("Age:", str(patient_details.get('age','N/A'))),
            ("Gender:", patient_details.get('gender','N/A')),
            ("Report Date:", pd.Timestamp.now().strftime("%d %b %Y %H:%M"))
        ]

        col1_label_x = left_margin + section_padding
        col1_value_x = 1.6 * inch
        col2_label_x = 4.0 * inch
        col2_value_x = 5.0 * inch
        label_value_gap = 0.1 * inch
        col_max_y = y_position

        for i, (label, value) in enumerate(details):
            is_first_column = (i % 2 == 0)
            x_label = col1_label_x if is_first_column else col2_label_x
            x_value = col1_value_x if is_first_column else col2_value_x
            if is_first_column:
                current_y = col_max_y

            pl = Paragraph(f"{label}", label_style)
            label_available_width = x_value - x_label - label_value_gap
            pl_w, pl_h = pl.wrapOn(c, label_available_width, 0.5 * inch)
            pl.drawOn(c, x_label, current_y - pl_h)

            available_width = (col2_label_x - col1_value_x - section_padding) if is_first_column else (right_margin - col2_value_x)
            pv_h = 0
            if available_width > 0.05 * inch:
                pv = Paragraph(value if value else "N/A", value_style)
                pv_w, pv_h = pv.wrapOn(c, available_width, 0.5 * inch)
                pv.drawOn(c, x_value, current_y - pv_h)
            row_min_y = current_y - max(pl_h, pv_h)
            if not is_first_column or i == len(details) - 1:
                col_max_y = row_min_y - 0.15 * inch
        y_position = col_max_y

        # ========== Symptoms Section ==========
        y_position -= section_padding
        c.setFillColor('#3498db')
        c.setStrokeColor('#3498db')
        c.roundRect(left_margin, y_position - section_header_height, right_margin - left_margin, section_header_height, radius=3, fill=1, stroke=1)
        p_symp_header = Paragraph("REPORTED SYMPTOMS", heading_style)
        p_w, p_h = p_symp_header.wrapOn(c, right_margin - left_margin - 2 * heading_style.leftPadding, section_header_height)
        p_symp_header.drawOn(c, left_margin + heading_style.leftPadding, y_position - p_h - (section_header_height - p_h)/2)
        y_position -= (section_header_height + 0.2 * inch)
        symptoms_text = ", ".join(patient_details.get('symptoms', ['N/A']))
        symptom_para = Paragraph(symptoms_text, body_style)
        symptom_w, symptom_h = symptom_para.wrapOn(c, right_margin - left_margin, 2 * inch)
        symptom_para.drawOn(c, left_margin, y_position - symptom_h)
        y_position -= (symptom_h + 0.3 * inch)

        # ========== Diagnosis Section ==========
        y_position -= section_padding
        c.setFillColor('#3498db')
        c.setStrokeColor('#3498db')
        c.roundRect(left_margin, y_position - section_header_height, right_margin - left_margin, section_header_height, radius=3, fill=1, stroke=1)
        p_diag_header = Paragraph("DIAGNOSIS RESULTS", heading_style)
        p_w, p_h = p_diag_header.wrapOn(c, right_margin - left_margin - 2 * heading_style.leftPadding, section_header_height)
        p_diag_header.drawOn(c, left_margin + heading_style.leftPadding, y_position - p_h - (section_header_height - p_h)/2)
        y_position -= (section_header_height + 0.3 * inch)

        models = [
            ("SVM Model", predictions.get('svm_model_prediction', 'N/A')),
            ("Naive Bayes", predictions.get('naive_bayes_prediction', 'N/A')),
            ("KNN Model", predictions.get('knn_model_prediction', 'N/A'))
        ]

        num_boxes = len(models)
        total_gap = (num_boxes - 1) * 0.25 * inch
        box_width = (right_margin - left_margin - total_gap) / num_boxes
        box_height = 0.6 * inch
        current_x = left_margin

        for model_name, prediction in models:
            c.setFillColor('#f8f9fa')
            c.setStrokeColor('#dee2e6')
            c.setLineWidth(1)
            c.roundRect(current_x, y_position - box_height, box_width, box_height, radius=3, fill=1, stroke=1)
            c.setFillColor('#3498db')
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(current_x + box_width/2, y_position - 0.2 * inch, model_name)
            c.setFillColor('#2c3e50')
            c.setFont("Helvetica-Bold", 10)
            c.drawCentredString(current_x + box_width/2, y_position - 0.45 * inch, prediction if prediction else "N/A")
            current_x += (box_width + 0.25 * inch)
        y_position -= (box_height + 0.3 * inch)

        # Final Diagnosis Box
        final_pred = predictions.get('final_prediction', 'N/A')
        final_box_height = 0.6 * inch
        c.setFillColor('#2ecc71')
        c.setStrokeColor('#2ecc71')
        c.roundRect(left_margin, y_position - final_box_height, right_margin - left_margin, final_box_height, radius=5, stroke=1, fill=1)
        c.setFillColor('#ffffff')
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(width / 2, y_position - 0.25 * inch, "FINAL DIAGNOSIS")
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, y_position - 0.5 * inch, final_pred if final_pred else "N/A")
        y_position -= (final_box_height + 0.3 * inch)

        # ========== Footer ==========
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor('#95a5a6')
        c.drawCentredString(width / 2, 0.5 * inch, "This is a system-generated report and does not require a signature.")

        c.save()
        logging.info("PDF report created successfully.")
    except Exception as e:
        logging.error(f"Failed to create PDF report: {e}")
# 6. Define Main Routes (Keep as before)
@app.route('/')
def index():
    session.pop('patient_details', None); session.pop('predictions', None); logging.info("Session cleared, rendering index.")
    if symptom_index_global: symptom_list_display = sorted(symptom_index_global.keys())
    else: symptom_list_display = ["Error loading symptoms"]; logging.error("Symptom index unavailable.")
    return render_template('index.html', symptoms=symptom_list_display)

@app.route('/result', methods=['POST'])
def result():
    logging.info("Handling POST /result route")
    if request.method == 'POST':
        symptoms_input = request.form.getlist('symptoms'); title = request.form.get('title', '').strip()
        first_name = request.form.get('first_name', '').strip(); last_name = request.form.get('last_name', '').strip()
        age_str = request.form.get('age', '').strip(); gender = request.form.get('gender', '')
        error_messages = [] # Input validation
        if not symptoms_input: error_messages.append("Select symptom(s).")
        if not first_name: error_messages.append("Enter first name.")
        if not last_name: error_messages.append("Enter last name.")
        if not age_str: error_messages.append("Enter age.")
        elif not age_str.isdigit() or not (0 < int(age_str) < 130): error_messages.append("Enter valid age.")
        if not gender: error_messages.append("Select gender.")
        if error_messages:
             logging.warning(f"Validation errors: {error_messages}")
             if symptom_index_global: symptom_list_display = sorted(symptom_index_global.keys())
             else: symptom_list_display = []
             return render_template('index.html', errors=error_messages, symptoms=symptom_list_display)
        session['patient_details'] = {'title': title, 'first_name': first_name, 'last_name': last_name, 'age': age_str, 'gender': gender, 'symptoms': symptoms_input}
        logging.debug(f"Patient details stored: {session['patient_details']}")
        try:
            if not all([symptom_index_global, encoder_global, svm_model_global, nb_model_global, knn_model_global, all_symptoms_list_global, X_train_global is not None, y_train_global is not None]):
                 logging.error("Models/data not ready."); return render_template('error.html', message="System error: Models not ready.")
            predictions = predict_disease(symptoms_input, symptom_index_global, encoder_global, svm_model_global, nb_model_global, knn_model_global, all_symptoms_list_global, X_train_global, y_train_global)
            session['predictions'] = predictions; logging.debug(f"Predictions stored: {session['predictions']}")
            return render_template('result.html', patient_details=session['patient_details'], predictions=predictions) # Pass both dicts
        except Exception as e: logging.exception("Prediction failed:"); return render_template('error.html', message=f"Prediction error: {e}")
    return redirect(url_for('index'))

@app.route('/view_report', methods=['GET'])
def view_report_get():
    logging.info("Handling GET /view_report route")
    patient_details = session.get('patient_details'); predictions = session.get('predictions')
    if not patient_details or not predictions:
        logging.warning("Session data missing for view_report. Redirecting.")
        return redirect(url_for('index'))
    logging.info("Session data retrieved for view_report.")
    return render_template('view_report.html', patient_details=patient_details, predictions=predictions)

@app.route('/download_report', methods=['POST'])
def download_report():
    logging.info("Handling POST /download_report route")
    try:
        patient_details = session.get('patient_details'); predictions = session.get('predictions')
        if not patient_details or not predictions: logging.error("Session data missing for PDF download."); return redirect(url_for('index'))
        logging.debug(f"Data for PDF: Patient={patient_details}, Pred={predictions}")
        hospital_logo_path = os.path.join(IMAGE_PATH, 'Hospital.png')
        if not os.path.exists(hospital_logo_path): logging.warning(f"Logo not found: {hospital_logo_path}."); hospital_logo_path = None
        hospital_details = {'name': 'AI Health Diagnostics Center', 'address': '123 Innovation Drive, Tech City', 'phone': '(555) 123-4567', 'email': 'contact@aihealthdiagnostics.com'}
        safe_first_name = "".join(c if c.isalnum() else "_" for c in patient_details.get('first_name', 'Report'))
        safe_last_name = "".join(c if c.isalnum() else "_" for c in patient_details.get('last_name', ''))
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"); filename = f"HealthReport_{safe_first_name}_{safe_last_name}_{timestamp}.pdf"
        filepath = os.path.join(REPORTS_PATH, filename)
        logging.info(f"Attempting PDF creation: {filepath}")
        create_pdf_report(filepath, copy.deepcopy(patient_details), copy.deepcopy(predictions), hospital_details, hospital_logo_path)
        logging.info(f"PDF created, sending: {filepath}")
        return send_file(filepath, as_attachment=True, download_name=filename)
    except FileNotFoundError as fnf_error: logging.error(f"PDF file not found post-creation: {fnf_error}"); return render_template('error.html', message="Error: Report file missing.")
    except Exception as e: logging.exception("Error during PDF download:"); return render_template('error.html', message="Error downloading report.")

# 7. Define Error Handlers (Keep as before)
@app.errorhandler(404)
def page_not_found(e): logging.error(f"404 Not Found: {request.path}"); return render_template('error.html', message="Page not found."), 404
@app.errorhandler(500)
def internal_server_error(e): logging.exception("500 Internal Server Error."); return render_template('error.html', message="Internal server error."), 500
@app.errorhandler(Exception)
def handle_exception(e): logging.exception(f"Unhandled exception: {e}"); return render_template('error.html', message="Unexpected error occurred."), 500

# 8. Application Startup Block (Keep as before)
if __name__ == '__main__':
    encoder_global = None; all_symptoms_list_global = None; svm_model_global = None
    nb_model_global = None; knn_model_global = None; symptom_index_global = None
    X_train_global = None; y_train_global = None
    try:
        X_train_global, X_test, y_train_global, y_test, encoder_global, all_symptoms_list_global = load_and_preprocess_data(DATA_PATH, TEST_DATA_PATH)
        svm_model_global, nb_model_global, knn_model_global = train_models(X_train_global, y_train_global)
        symptom_index_global = create_symptom_index(all_symptoms_list_global)
        if not all([symptom_index_global, encoder_global, svm_model_global, nb_model_global, knn_model_global]): raise ValueError("Model/data loading failed.")
        logging.info("Application initialized successfully.")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except FileNotFoundError as startup_fnf: logging.critical(f"FATAL: Data file missing: {startup_fnf}. Exiting.", exc_info=True); print(f"FATAL ERROR: Data file missing: {startup_fnf}. Check paths."); exit(1)
    except Exception as startup_error: logging.critical(f"FATAL: App startup failed: {startup_error}", exc_info=True); print(f"\nFATAL STARTUP ERROR: {startup_error}\nCheck logs."); exit(1)