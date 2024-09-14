import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # to ignore some kind of error


import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # cool library for similarity search
import streamlit as st  # web app

# libraries needed for processing
from transformers import pipeline
import PyPDF2
import torch
import warnings

# libraries for data management
import sqlite3  
import hashlib  
import uuid  
from datetime import datetime, timedelta

# libraries for data visualization in dashboard
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# set streamlit config at top to avoid error
st.set_page_config(page_title="HealthSim: Advanced Patient Matching System", layout="wide")

# I always get this error coming up in terminal...
warnings.simplefilter(action='ignore', category=FutureWarning)

# load transformer model for matching
@st.cache_resource  # Cache the model to avoid reloading
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_embedding_model()

# Function to load and process the doctor database
@st.cache_data  # cache database to not take so long each time
def load_doctor_database(json_file_path):
    with open(json_file_path, 'r') as file:
        doctor_data = json.load(file)
    doctor_profiles = []
    doctor_texts = []

    for doctor in doctor_data:
        # making an outline for each doctor profile
        profile_text = (
            f"Name: {doctor['name']}. "
            f"Specialty: {doctor['specialty']}. "
            f"Experience: {doctor['experience']}. "
            f"Availability: {doctor['availability']}. "
            f"Past Cases: {', '.join(doctor['past_cases'])}. "
            f"Location: {doctor['location']}. "
            f"Languages: {', '.join(doctor['languages'])}. "
            f"Lifestyle Specialty: {doctor['lifestyle_specialty']}. "
            f"Demographics: {doctor['demographics']}. "
            f"Lifestyle Approach: {doctor['lifestyle_approach']}. "
            f"Success Rate: {doctor['success_rate']}. "
            f"Offers Counseling: {'Yes' if doctor['offers_counseling'] else 'No'}. "
            f"Education: {doctor['education_certifications']['medical_school']}, "
            f"Residency at {doctor['education_certifications']['residency']}, "
            f"Fellowship in {doctor['education_certifications']['fellowship']}, "
            f"Board Certifications: {', '.join(doctor['education_certifications']['board_certifications'])}. "
            f"Research Publications: {'; '.join([pub['title'] for pub in doctor['research_publications']])}. "
            f"Treatment Philosophy: {doctor['treatment_philosophy']}. "
            f"Technological Expertise: {', '.join(doctor['technological_expertise'])}. "
            f"Patient Reviews: {doctor['patient_reviews']['feedback_summary']} "
            f"with Rating: {doctor['patient_reviews']['rating']}. "
            f"Community Involvement: {', '.join(doctor['community_involvement'])}. "
            f"Insurance Accepted: {', '.join(doctor['insurance_accepted'])}. "
            f"Accessibility Features: {', '.join(doctor['accessibility_features'])}."
        )
        doctor_profiles.append(doctor)
        doctor_texts.append(profile_text)

    # get embeddings of each doctor
    embeddings = model.encode(doctor_texts, convert_to_tensor=False)

    # make into numpy array to work with
    embeddings = np.array(embeddings).astype('float32')

    # Make FAISS index to work with search function
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product (dot product) to get similarity
    faiss.normalize_L2(embeddings)  # normalize the embeddings
    index.add(embeddings)

    return doctor_profiles, index, embeddings

# function to find top k doctors that match based on FAISS search
def find_best_matches(patient_input_text, doctor_profiles, index, top_k=5):
    # get embedding of patient input (form data)
    patient_embedding = model.encode([patient_input_text], convert_to_tensor=False)
    patient_embedding = np.array(patient_embedding).astype('float32')
    faiss.normalize_L2(patient_embedding)  # normalize the embedding

    # search for the top k doctors (5 in this case)
    D, I = index.search(patient_embedding, top_k)
    matches = [(doctor_profiles[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return matches

# function to process EHR data or other files that are put in
def process_ehr_data(ehr_file):
    ehr_text = ""
    if ehr_file is not None:
        if ehr_file.type == 'text/plain':
            ehr_text = ehr_file.getvalue().decode("utf-8")
        elif ehr_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(ehr_file)
            for page in pdf_reader.pages:
                ehr_text += page.extract_text()
        else:
            st.error("Unsupported file type.")
            return ""
        # Bart Large CNN to get info and summarize it
        try:
            # Determine device
            device = 0 if torch.cuda.is_available() else -1
            summarizer = pipeline(
                "summarization",
                model='facebook/bart-large-cnn',
                device=device
            )
            input_length = len(ehr_text.split())
            max_length = min(512, int(input_length * 0.3)) # set max length of output compared to input
            max_length = max(50, max_length)  # set min length of output
            summary_list = summarizer(
                ehr_text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                clean_up_tokenization_spaces=True  # gotta get rid of annoying future warning error
            )
            ehr_summary = summary_list[0]['summary_text']
            return ehr_summary
        except Exception as e:
            st.error(f"Error processing EHR data: {str(e)}")
            return ""
    return ""

# initialize our db
def init_db():
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    # make tables
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # make appointment tables
    c.execute('''CREATE TABLE IF NOT EXISTS appointments
                 (id TEXT PRIMARY KEY, username TEXT, patient_name TEXT, appointment_date TEXT, appointment_time TEXT,
                  doctor TEXT, notes TEXT)''')
    # make patient tables
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (patient_id TEXT PRIMARY KEY, username TEXT, name TEXT, age INTEGER, weight REAL,
                  conditions TEXT, ehr_summary TEXT)''')
    conn.commit()
    conn.close()

# fash passwords for safety
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# password verification of user
def verify_password(password, hashed):
    return hash_password(password) == hashed

# adding user to database
def add_user(username, password):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
    conn.commit()
    conn.close()

# check if user exists and also verify their password in login
def login_user(username, password):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return verify_password(password, data[0])
    return False

# add appointment to database
def add_appointment(username, appointment):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('''INSERT INTO appointments (id, username, patient_name, appointment_date, appointment_time, doctor, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (str(uuid.uuid4()), username, appointment['patient_name'], appointment['appointment_date'],
               appointment['appointment_time'], appointment['doctor'], appointment['notes']))
    conn.commit()
    conn.close()

# get the appointments for the user
def get_appointments(username):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('SELECT id, patient_name, appointment_date, appointment_time, doctor, notes FROM appointments WHERE username = ?',
              (username,))
    data = c.fetchall()
    conn.close()
    return data

# add in the patient record data
def add_patient(username, patient):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('''INSERT INTO patients (patient_id, username, name, age, weight, conditions, ehr_summary)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (str(uuid.uuid4()), username, patient['name'], patient['age'], patient['weight'],
               patient['conditions'], patient['ehr_summary']))
    conn.commit()
    conn.close()

# get the patients of the given user
def get_patients(username):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('SELECT patient_id, name, age, weight, conditions, ehr_summary FROM patients WHERE username = ?', (username,))
    data = c.fetchall()
    conn.close()
    return data

# delete an appointment
def delete_appointment(appointment_id):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('DELETE FROM appointments WHERE id = ?', (appointment_id,))
    conn.commit()
    conn.close()

# update an appointment to edit data
def update_appointment(appointment_id, appointment):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('''UPDATE appointments SET patient_name=?, appointment_date=?, appointment_time=?, doctor=?, notes=?
                 WHERE id=?''',
              (appointment['patient_name'], appointment['appointment_date'], appointment['appointment_time'],
               appointment['doctor'], appointment['notes'], appointment_id))
    conn.commit()
    conn.close()

# delete a patient from database
def delete_patient(patient_id):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
    conn.commit()
    conn.close()

# update/edit patient info
def update_patient(patient_id, patient):
    conn = sqlite3.connect('healthsim.db')
    c = conn.cursor()
    c.execute('''UPDATE patients SET name=?, age=?, weight=?, conditions=?, ehr_summary=?
                 WHERE patient_id=?''',
              (patient['name'], patient['age'], patient['weight'], patient['conditions'], patient['ehr_summary'], patient_id))
    conn.commit()
    conn.close()


init_db()

# initiliaize session state for authentication and other stuffI
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'selected_doctor' not in st.session_state:
    st.session_state.selected_doctor = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ''
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = ''

# main app with streamlit
def main():
    # some css code
    st.markdown(
        """
        <style>
        /* Style for the main content */
        .main {
            background-color: #f0f2f6;
            padding: 20px;
        }

        /* Style for buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            margin-top: 10px;
        }

        /* Style for sliders */
        .stSlider > div {
            padding-top: 20px;
        }

        /* Style for matched doctor cards */
        .doctor-card {
            background-color: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .doctor-card h3 {
            margin-top: 0;
        }

        /* Style for sidebar */
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🩺 HealthSim: Advanced Patient Matching System")

    # user authentication
    if not st.session_state.logged_in:
        st.sidebar.header("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Logged in as {username}")
                st.experimental_rerun()  # go back to login state after logged in
            else:
                st.error("Invalid username or password")
        st.sidebar.write("Don't have an account? Sign up below.")
        signup_username = st.sidebar.text_input("New Username")
        signup_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign Up"):
            try:
                add_user(signup_username, signup_password)
                st.success("User created successfully! Please log in.")
            except sqlite3.IntegrityError:
                st.error("Username already exists")
    else:
        # load up doctor db
        with st.spinner('Loading doctor database...'):
            doctor_profiles, index, embeddings = load_doctor_database('enhanced_specialists_large.json')

        # set up sidebar for nav with different pages
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", ["Home", "Appointments", "Patients", "Dashboard", "Reports", "About", "Logout"])

        if selection == "Home":
            st.session_state.page = 'home'  # starting page is home
            if st.session_state.selected_doctor:
                # setting appointment with doctor in matching result screen
                doctor = st.session_state.selected_doctor
                patient_name = st.session_state.patient_name
                st.header(f"Schedule Appointment with {doctor['name']}")
                with st.form("schedule_form"):
                    appointment_date = st.date_input("Appointment Date", datetime.today())
                    appointment_time = st.time_input("Appointment Time", datetime.now().time())
                    notes = st.text_area("Notes")
                    submitted = st.form_submit_button("Schedule Appointment")
                if submitted:
                    appointment = {
                        'patient_name': patient_name,
                        'appointment_date': str(appointment_date),
                        'appointment_time': str(appointment_time),
                        'doctor': doctor['name'],
                        'notes': notes
                    }
                    add_appointment(st.session_state.username, appointment)
                    st.success("Appointment scheduled successfully!")
                    # when done, clear doctor name and patient name
                    st.session_state.selected_doctor = None
                    st.session_state.patient_name = ''
                    st.experimental_rerun()
            else:
                # form for PCP to manually input
                st.header("Enter Patient Information")
                with st.form("patient_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        patient_name = st.text_input("Patient Name")
                        location = st.selectbox("Location", ["Los Angeles", "San Francisco", "Atlanta", "Chicago"])
                        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
                        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
                        lifestyle_choices = st.selectbox("Lifestyle Choices", ["Active", "Sedentary", "Moderate"])
                        smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
                        alcohol_consumption = st.slider("Alcohol Consumption (units per week)", 0, 50, 0)
                        # place to upload additional documents like EHR data for CNN model to read
                        ehr_file = st.file_uploader("Upload EHR Data (optional)", type=["txt", "pdf"])

                    with col2:
                        family_history_options = ["Heart Disease", "Diabetes", "Cancer", "Hypertension", "Asthma",  "None"]
                        family_history = st.multiselect("Family History (select all that apply)", family_history_options)
                        family_history_details = st.text_area("Additional Family History Details")
                        current_symptoms = st.text_area("Current Symptoms")
                        symptom_severity = st.slider("Symptom Severity", 0, 10, 5)
                        symptom_duration = st.number_input("Symptom Duration (days)", min_value=0, value=1, step=1)
                        condition = st.text_input("Known Medical Conditions")

                    submitted = st.form_submit_button("🔍 Find Best Matches")

                if submitted:
                    # make sure there are actual inputs
                    if (not current_symptoms.strip() and not condition.strip()) and ehr_file is None:
                        st.warning("Please enter current symptoms, known medical conditions, or upload EHR data.")
                    else:
                        with st.spinner('Processing...'):
                            # summarize the ehr data to feed into RAG model
                            ehr_summary = process_ehr_data(ehr_file)

                            # easy showcase of patient data in one piece (anime ref)
                            patient_input_text = (
                                f"Location: {location}. "
                                f"Age: {age}. "
                                f"Weight: {weight} kg. "
                                f"Lifestyle: {lifestyle_choices}. "
                                f"Smoker: {smoke}. "
                                f"Alcohol Consumption: {alcohol_consumption} units/week. "
                                f"Family History: {', '.join(family_history)}. "
                                f"Additional Family History Details: {family_history_details}. "
                                f"Current Symptoms: {current_symptoms}. "
                                f"Symptom Severity: {symptom_severity}/10. "
                                f"Symptom Duration: {symptom_duration} days. "
                                f"Known Conditions: {condition}. "
                                f"EHR Summary: {ehr_summary}"
                            )

                            # RAG model to get best matches based on patient data and similarity to doctor profiles
                            matches = find_best_matches(patient_input_text, doctor_profiles, index)

                            # save the patient data (very concise)
                            patient_data = {
                                'name': patient_name,
                                'age': age,
                                'weight': weight,
                                'conditions': condition,
                                'ehr_summary': ehr_summary
                            }
                            add_patient(st.session_state.username, patient_data)
                            st.success("Patient record saved.")

                        # show the matched doctors (5 in this case)
                        st.header("Best Matched Doctors:")
                        if not matches:
                            st.info("No doctors match the given criteria.")
                        else:
                            for idx, (doctor, score) in enumerate(matches):
                                # make a card for each doctor
                                st.markdown(
                                    f"""
                                    <div class="doctor-card">
                                        <h3>{doctor['name']} (Score: {score:.2f})</h3>
                                        <p><strong>Specialty:</strong> {doctor['specialty']}</p>
                                        <p><strong>Experience:</strong> {doctor['experience']}</p>
                                        <p><strong>Availability:</strong> {doctor['availability']}</p>
                                        <p><strong>Location:</strong> {doctor['location']}</p>
                                        <p><strong>Languages:</strong> {', '.join(doctor['languages'])}</p>
                                        <p><strong>Success Rate:</strong> {doctor['success_rate']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # show full info here if the user wants to see more
                                with st.expander("See more details"):
                                    st.write(f"Patient Reviews: {doctor['patient_reviews']['feedback_summary']} (Rating: {doctor['patient_reviews']['rating']})")
                                    st.write(f"Education: {doctor['education_certifications']['medical_school']}, Residency at {doctor['education_certifications']['residency']}, Fellowship in {doctor['education_certifications']['fellowship']}, Board Certifications: {', '.join(doctor['education_certifications']['board_certifications'])}")
                                    st.write(f"Research Publications: {'; '.join([pub['title'] for pub in doctor['research_publications']])}")
                                    st.write(f"Treatment Philosophy: {doctor['treatment_philosophy']}")
                                    st.write(f"Technological Expertise: {', '.join(doctor['technological_expertise'])}")
                                    st.write(f"Community Involvement: {', '.join(doctor['community_involvement'])}")
                                    st.write(f"Insurance Accepted: {', '.join(doctor['insurance_accepted'])}")
                                    st.write(f"Accessibility Features: {', '.join(doctor['accessibility_features'])}")
                                if st.button(f"Schedule Appointment with {doctor['name']}", key=f"schedule_{idx}"):
                                    st.session_state.selected_doctor = doctor
                                    st.session_state.patient_name = patient_name
                                    st.experimental_rerun()

        elif selection == "Appointments":
            st.header("Manage Appointments")

            # make more appointments with existing patients already in db
            st.subheader("Add New Appointment")
            with st.form("new_appointment_form"):
                patients = get_patients(st.session_state.username)
                patient_names = [patient[1] for patient in patients]
                if not patient_names:
                    st.warning("No patients found. Please add patients first.")
                else:
                    selected_patient = st.selectbox("Select Patient", patient_names)
                    appointment_date = st.date_input("Appointment Date", datetime.today())
                    appointment_time = st.time_input("Appointment Time", datetime.now().time())
                    # Select a doctor
                    doctor_names = [doctor['name'] for doctor in doctor_profiles]
                    selected_doctor = st.selectbox("Select Doctor", doctor_names)
                    notes = st.text_area("Notes")
                    submitted = st.form_submit_button("Add Appointment")
                if submitted:
                    appointment = {
                        'patient_name': selected_patient,
                        'appointment_date': str(appointment_date),
                        'appointment_time': str(appointment_time),
                        'doctor': selected_doctor,
                        'notes': notes
                    }
                    add_appointment(st.session_state.username, appointment)
                    st.success("Appointment added successfully!")
                    st.experimental_rerun()

            # show the existing appointments
            st.subheader("Existing Appointments")
            appointments = get_appointments(st.session_state.username)
            if not appointments:
                st.info("No appointments scheduled.")
            else:
                for appointment in appointments:
                    st.markdown(
                        f"""
                        <div class="doctor-card">
                            <h3>{appointment[1]} with {appointment[4]}</h3>
                            <p><strong>Date:</strong> {appointment[2]}</p>
                            <p><strong>Time:</strong> {appointment[3]}</p>
                            <p><strong>Notes:</strong> {appointment[5]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # ability to delete appointment if necessary
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Edit", key=f"edit_{appointment[0]}"):
                            st.session_state['edit_appointment_id'] = appointment[0]
                            st.session_state['edit_appointment_data'] = appointment
                    with col2:
                        if st.button("Delete", key=f"delete_{appointment[0]}"):
                            delete_appointment(appointment[0])
                            st.success("Appointment deleted.")
                            st.experimental_rerun()
                # ability to edit appointment if necessary
                if 'edit_appointment_id' in st.session_state:
                    appointment_id = st.session_state['edit_appointment_id']
                    appointment_data = st.session_state['edit_appointment_data']
                    st.header("Edit Appointment")
                    with st.form("edit_appointment_form"):
                        patient_name = st.text_input("Patient Name", value=appointment_data[1])
                        appointment_date = st.date_input("Appointment Date", value=datetime.strptime(appointment_data[2], '%Y-%m-%d'))
                        appointment_time = st.time_input("Appointment Time", value=datetime.strptime(appointment_data[3], '%H:%M:%S').time())
                        doctor_name = st.text_input("Doctor", value=appointment_data[4])
                        notes = st.text_area("Notes", value=appointment_data[5])
                        submitted = st.form_submit_button("Update Appointment")
                    if submitted:
                        appointment = {
                            'patient_name': patient_name,
                            'appointment_date': str(appointment_date),
                            'appointment_time': str(appointment_time),
                            'doctor': doctor_name,
                            'notes': notes
                        }
                        update_appointment(appointment_id, appointment)
                        st.success("Appointment updated successfully!")
                        del st.session_state['edit_appointment_id']
                        del st.session_state['edit_appointment_data']
                        st.experimental_rerun()

        elif selection == "Patients":
            st.header("Patient Records")
            patients = get_patients(st.session_state.username)
            if not patients:
                st.info("No patient records found.")
            else:
                for patient in patients:
                    st.markdown(
                        f"""
                        <div class="doctor-card">
                            <h3>{patient[1]}</h3>
                            <p><strong>Age:</strong> {patient[2]}</p>
                            <p><strong>Weight:</strong> {patient[3]} kg</p>
                            <p><strong>Conditions:</strong> {patient[4]}</p>
                            <p><strong>EHR Summary:</strong> {patient[5]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # ability to edit or delete the patient from db
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Edit", key=f"edit_patient_{patient[0]}"):
                            st.session_state['edit_patient_id'] = patient[0]
                            st.session_state['edit_patient_data'] = patient
                    with col2:
                        if st.button("Delete", key=f"delete_patient_{patient[0]}"):
                            delete_patient(patient[0])
                            st.success("Patient record deleted.")
                            st.experimental_rerun()
                # edit the patient info
                if 'edit_patient_id' in st.session_state:
                    patient_id = st.session_state['edit_patient_id']
                    patient_data = st.session_state['edit_patient_data']
                    st.header("Edit Patient Record")
                    with st.form("edit_patient_form"):
                        name = st.text_input("Patient Name", value=patient_data[1])
                        age = st.number_input("Age", min_value=0, max_value=120, value=patient_data[2], step=1)
                        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=patient_data[3], step=0.1)
                        conditions = st.text_input("Known Medical Conditions", value=patient_data[4])
                        ehr_summary = st.text_area("EHR Summary", value=patient_data[5])
                        submitted = st.form_submit_button("Update Patient")
                    if submitted:
                        patient = {
                            'name': name,
                            'age': age,
                            'weight': weight,
                            'conditions': conditions,
                            'ehr_summary': ehr_summary
                        }
                        update_patient(patient_id, patient)
                        st.success("Patient record updated successfully!")
                        del st.session_state['edit_patient_id']
                        del st.session_state['edit_patient_data']
                        st.experimental_rerun()

        elif selection == "Dashboard":
            st.header("📊 Dashboard")
            # get data to show in data visualization stuff like graphs
            appointments = get_appointments(st.session_state.username)
            patients = get_patients(st.session_state.username)
            num_appointments = len(appointments)
            num_patients = len(patients)

            upcoming_appointments = [appt for appt in appointments if datetime.strptime(appt[2], '%Y-%m-%d') >= datetime.today()]
            num_upcoming_appointments = len(upcoming_appointments)

            # Display statistics with metrics
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Patients", num_patients)
            col2.metric("Total Appointments", num_appointments)
            col3.metric("Upcoming Appointments", num_upcoming_appointments)

            # setting up data for visualization
            if patients:
                patients_df = pd.DataFrame(patients, columns=['patient_id', 'Name', 'Age', 'Weight', 'Conditions', 'EHR_Summary'])
                patients_df['Age'] = patients_df['Age'].astype(int)
                patients_df['Weight'] = patients_df['Weight'].astype(float)
            else:
                patients_df = pd.DataFrame(columns=['patient_id', 'Name', 'Age', 'Weight', 'Conditions', 'EHR_Summary'])

            if appointments:
                appointments_df = pd.DataFrame(appointments, columns=['id', 'Patient_Name', 'Date', 'Time', 'Doctor', 'Notes'])
                appointments_df['Date'] = pd.to_datetime(appointments_df['Date'])
            else:
                appointments_df = pd.DataFrame(columns=['id', 'Patient_Name', 'Date', 'Time', 'Doctor', 'Notes'])

            # cool plots
            st.subheader("Patient Demographics")
            col1, col2 = st.columns(2)

            with col1:
                # showing distribution of patients' age
                if not patients_df.empty:
                    fig, ax = plt.subplots()
                    sns.histplot(patients_df['Age'], bins=10, kde=True, color='skyblue', ax=ax)
                    ax.set_title('Age Distribution of Patients')
                    st.pyplot(fig)
                else:
                    st.info("No patient data available for plotting.")

            with col2:
                # showing distribution of patients' weight
                if not patients_df.empty:
                    fig, ax = plt.subplots()
                    sns.histplot(patients_df['Weight'], bins=10, kde=True, color='lightgreen', ax=ax)
                    ax.set_title('Weight Distribution of Patients')
                    st.pyplot(fig)
                else:
                    st.info("No patient data available for plotting.")

            st.subheader("Appointments Over Time")
            if not appointments_df.empty:
                # showing number of appointments daily
                appointments_count = appointments_df.groupby(appointments_df['Date'].dt.date).size().reset_index(name='Appointments')
                fig, ax = plt.subplots()
                sns.lineplot(x='Date', y='Appointments', data=appointments_count, marker='o', ax=ax)
                ax.set_title('Number of Appointments Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Appointments')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No appointment data available for plotting.")

            # showing appointments in a cool table that can be downloaded
            st.subheader("Upcoming Appointments")
            if not upcoming_appointments:
                st.info("No upcoming appointments.")
            else:
                upcoming_df = pd.DataFrame(upcoming_appointments, columns=['id', 'Patient_Name', 'Date', 'Time', 'Doctor', 'Notes'])
                upcoming_df['Date'] = pd.to_datetime(upcoming_df['Date'])
                upcoming_df = upcoming_df.sort_values('Date')
                st.dataframe(upcoming_df[['Patient_Name', 'Date', 'Time', 'Doctor', 'Notes']])

        elif selection == "Reports":
            st.header("📄 Generate Patient Reports")
            patients = get_patients(st.session_state.username)
            if not patients:
                st.info("No patient records found.")
            else:
                patient_names = [patient[1] for patient in patients]
                selected_patient = st.selectbox("Select Patient", patient_names)
                if st.button("Generate Report"):
                    patient_data = next((p for p in patients if p[1] == selected_patient), None)
                    if patient_data:
                        st.subheader(f"Report for {patient_data[1]}")
                        st.write(f"**Age:** {patient_data[2]}")
                        st.write(f"**Weight:** {patient_data[3]} kg")
                        st.write(f"**Conditions:** {patient_data[4]}")
                        st.write(f"**EHR Summary:** {patient_data[5]}")

                        # can download patient info as a txt file
                        report_content = f"""
                        Report for {patient_data[1]}
                        Age: {patient_data[2]}
                        Weight: {patient_data[3]} kg
                        Conditions: {patient_data[4]}
                        EHR Summary: {patient_data[5]}
                        """
                        st.download_button(
                            label="Download Report",
                            data=report_content,
                            file_name=f"{patient_data[1]}_report.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Patient data not found.")

        elif selection == "About":
            st.header("About HealthSim")
            st.write("""
                **HealthSim** is a product developed by Preetham Manapuri, Neil Raman, Jack London, and Anthony Xu
            """)

            st.subheader("Key Features")
            st.markdown("""
            - **In Progress**
            """)

        elif selection == "Logout": # logout feature to go back to login screen
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.session_state.selected_doctor = None
            st.session_state.patient_name = ''
            st.session_state.page = 'home'
            st.success("You have been logged out.")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
