import streamlit as st
from pymongo import MongoClient
import bcrypt
import joblib
import numpy as np
import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3

import re

# email validation function
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    return re.match(pattern, email)


st.set_page_config(page_title="AI Pneumonia Detection", layout="wide")

# LOAD MODEL 
model = joblib.load("pneumonia_model.pkl")
xray_model = tf.keras.models.load_model("xray_model.keras", compile=False)

# SESSION FILE
SESSION_FILE = "session.json"

if os.path.exists(SESSION_FILE):
    with open(SESSION_FILE, "r") as f:
        data = json.load(f)
        st.session_state.logged_in = data.get("logged_in", False)
        st.session_state.user_name = data.get("user_name", "")
else:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""

if "page" not in st.session_state:
    st.session_state.page = "home"

# CSS
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Database 
MONGO_URI = "mongodb+srv://tusharwalia24:Harry%4098147@aiproject.eveznxp.mongodb.net/?appName=AiProject"
client = MongoClient(MONGO_URI)
db = client["healthcare_db"]
users_collection = db["users"]
history_collection = db["history"]

st.title("🏥 AI Pneumonia Detection System")


# Chatbot

genai.configure(api_key="AIzaSyA-qNq3tsd_9qYtbv0bXmIjRf2aYJqNYxs")
model_ai = genai.GenerativeModel("gemini-2.5-flash")

def health_chatbot(msg):

    prompt = f"""
You are healthcare assistant.
Answer only health related questions.
If not health question say: I only answer health questions.

Question: {msg}
"""

    response = model_ai.generate_content(prompt)
    return response.text

# PDF Creation

def create_pdf(user, age, result):

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter

# HEADER 
    c.setFillColor(colors.darkblue)
    c.rect(0, height-80, width, 80, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(150, height-50, "AI HEALTHCARE REPORT")

# TITLE 
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, height-120, "Pneumonia Prediction Report")

# PATIENT BOX 
    c.setStrokeColor(colors.black)
    c.rect(50, height-250, width-100, 100, fill=0)

    c.setFont("Helvetica", 12)
    c.drawString(70, height-180, f"Patient Name : {user}")
    c.drawString(70, height-210, f"Age : {age}")

# RESULT BOX 
    if "High" in result:
        c.setFillColor(colors.red)
    else:
        c.setFillColor(colors.green)

    c.rect(50, height-350, width-100, 60, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(70, height-320, f"Prediction Result : {result}")

# FOOTER
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(50, 100, "This is an AI generated report.")
    c.drawString(50, 80, "Consult a doctor for professional medical advice.")

    c.drawString(50, 50, "AI Healthcare System © 2026")

    c.save()
    buffer.seek(0)
    return buffer


def voice_assistant():

    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty('rate',170)

    st.info("🎤 Interactive voice started. Say 'stop' to exit.")

    while True:

        try:
# Listening
            with sr.Microphone() as source:
                st.write("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, phrase_time_limit=5)

            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")

# To Stop The Voice Assitant
            if "stop" in text.lower() or "exit" in text.lower():
                engine.say("Goodbye")
                engine.runAndWait()
                break

# Answer 
            prompt = f"Give short healthcare answer: {text}"
            response = model_ai.generate_content(prompt)
            reply = response.text

            st.write("AI:", reply)

# SPEAK
            engine.say(reply)
            engine.runAndWait()

            import time
            time.sleep(5)

        except sr.UnknownValueError:
            st.warning("Speak clearly...")

        except Exception as e:
            st.error("Voice assistant stopped")
            break

# After Login Bar
if st.session_state.logged_in:

    st.sidebar.title(f"👋 {st.session_state.user_name}")

    if st.sidebar.button("🏠 Dashboard"):
        st.session_state.page = "dashboard"

    if st.sidebar.button("🩻 X-ray Prediction"):
        st.session_state.page = "xray"
    
    if st.sidebar.button("🎤 Voice Assistant"):
        st.session_state.page = "voice"

    if st.sidebar.button("💬 Health Chatbot"):
        st.session_state.page = "chatbot"

    if st.sidebar.button("🧠 Prediction"):
        st.session_state.page = "predict"

    if st.sidebar.button("📊 History"):
        st.session_state.page = "history"

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
        st.session_state.page = "home"
        st.rerun()

# Before Login Navbar
else:
    st.sidebar.title("Navigation")

    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "home"

    if st.sidebar.button("🔑 Login"):
        st.session_state.page = "login"

    if st.sidebar.button("📝 Sign Up"):
        st.session_state.page = "signup"

# Home Page
if st.session_state.page == "home":
    st.header("Welcome to AI Pneumonia Detection System")
    st.write("Login to use AI prediction features.")

# Sign Up Page
elif st.session_state.page == "signup":
    st.header("Create Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):

        if not is_valid_email(email):
            st.error("Enter valid Gmail (example@gmail.com)")

        elif len(password) < 4:
            st.error("Password must be at least 4 characters")

        elif users_collection.find_one({"email": email}):
            st.warning("Email already exists")

        else:
            hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            users_collection.insert_one({
                "name": name,
                "email": email,
                "password": hashed_pw
            })

            st.session_state.logged_in = True
            st.session_state.user_name = name

            with open(SESSION_FILE, "w") as f:
                json.dump({"logged_in": True, "user_name": name}, f)

            st.success("Account created & logged in")
            st.rerun()

# Login
elif st.session_state.page == "login":
    st.header("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

# Email Validation
        if not is_valid_email(email):
            st.error("Enter valid Gmail address")

        else:
            user = users_collection.find_one({"email": email})

            if user:
                stored_password = user["password"]

                if isinstance(stored_password, str):
                    stored_password = stored_password.encode()

                if bcrypt.checkpw(password.encode(), stored_password):
                    st.session_state.logged_in = True
                    st.session_state.user_name = user["name"]

                    with open(SESSION_FILE, "w") as f:
                        json.dump({"logged_in": True, "user_name": user["name"]}, f)

                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error("Wrong password")
            else:
                st.error("User not found")

# Dasboard
elif st.session_state.page == "dashboard":
    st.header("Dashboard")
    st.success("AI model connected successfully")
    st.subheader("This is the dashboard of this project")

# Prediction 
elif st.session_state.page == "predict":

    st.header("🧠 Pneumonia Prediction")
    st.subheader("Select symptoms")

    fever = int(st.checkbox("Fever"))
    cough = int(st.checkbox("Cough"))
    breath = int(st.checkbox("Breathing Problem"))
    chest = int(st.checkbox("Chest Pain"))
    fatigue = int(st.checkbox("Fatigue"))
    throat = int(st.checkbox("Sore Throat"))
    body = int(st.checkbox("Body Ache"))
    smoking = int(st.checkbox("Smoking"))
    diabetes = int(st.checkbox("Diabetes"))

    age = st.number_input("Enter Age", 1, 100)

    if st.button("🔍 Predict"):

        input_data = np.array([[fever,cough,breath,chest,fatigue,
                                throat,body,smoking,diabetes,age]])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.divider()

        if prediction == 1:
            result_text = "High chance of Pneumonia"
            st.error(f"⚠ High chance ({prob[1]*100:.2f}% risk)")
        else:
            result_text = "Low chance of Pneumonia"
            st.success(f"✅ Low chance ({prob[0]*100:.2f}% safe)")

        # SAVE HISTORY
        history_collection.insert_one({
            "user": st.session_state.user_name,
            "age": age,
            "result": result_text
        })

# PDF Download Medical Report
        pdf_buffer = create_pdf(st.session_state.user_name, age, result_text)

        st.download_button(
            label="📄 Download Medical Report",
            data=pdf_buffer,
            file_name="Pneumonia_Report.pdf",
            mime="application/pdf"
        )

# History Page
elif st.session_state.page == "history":

    st.header("📊 Recent Predictions")

    records = history_collection.find(
        {"user": st.session_state.user_name}
    ).sort("_id", -1).limit(10)

    found = False

    for r in records:
        found = True
        st.write("👤 User:", r["user"])
        st.write("🎂 Age:", r["age"])
        st.write("🩺 Result:", r["result"])
        st.write("------")

    if not found:
        st.info("No history found")

# XRAY PREDICTION PAGE 

elif st.session_state.page == "xray":

    st.header("🩻 X-ray Pneumonia Detection")
    st.info("Upload Chest X-ray image for prediction")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", width=300)

        if st.button("🔍 Predict"):

            # resize exactly like training
            img = image.resize((224,224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = xray_model.predict(img)[0][0]

            st.write(f"Model confidence score: {pred:.4f}")

            if pred > 0.5:
                st.error(f"⚠ Pneumonia Detected ({pred*100:.2f}% confidence)")
            else:
                st.success(f"✅ Normal ({(1-pred)*100:.2f}% confidence)")

# VOICE PAGE 
elif st.session_state.page == "voice":

    st.header("🎤 AI Voice Health Assistant")

    st.write("Click button and speak your health question")

    if st.button("Start Voice Assistant"):
        voice_assistant()

# CHATBOT PAGE

elif st.session_state.page == "chatbot":

    st.header("💬 AI Health Assistant")

    user_input = st.text_input("Ask health question")

    if st.button("Ask AI"):

        if user_input.strip() == "":
            st.warning("Type something")

        else:
        
            prompt = f"""
You are a friendly healthcare assistant.

Explain in VERY SIMPLE language.
Explain like talking to a normal person.
Use short sentences.
Avoid difficult medical words.
Keep answer under 3 lines.
Motivate the user. Also just give them health related answers no general questions please

Question: {user_input}
"""

            response = model_ai.generate_content(prompt)
            reply = response.text

            st.success(reply)