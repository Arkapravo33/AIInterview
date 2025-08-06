import cv2
import numpy as np
import speech_recognition as sr
import openai
import time
import threading
import datetime
import os
import csv
import librosa
import soundfile as sf

# ------------------ CONFIGURATION ------------------
openai.api_key = "sk-REPLACE_WITH_YOUR_OPENAI_KEY"

questions = [
    "Tell me about yourself.",
    "Why should we hire you?",
    "What are your strengths and weaknesses?",
    "Where do you see yourself in 5 years?"
]

output_file = "interview_log.csv"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------ GLOBAL VARIABLES ------------------
cap = None
current_frame = None
camera_running = False
log_data = []

# ------------------ CAMERA SETUP ------------------
def initialize_camera():
    global cap, camera_running
    print("\U0001F3A5 Initializing camera...")
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return False

    camera_running = True
    return True

def camera_monitor():
    global cap, current_frame, camera_running
    while camera_running:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                current_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, "Interview in Progress", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Interview Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\U0001F44B User exited camera.")
                    camera_running = False
                    break
        time.sleep(0.03)

def wait_for_face(timeout=30):
    global current_frame
    print("➡ Please look at the camera. Waiting for face...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if current_frame is not None:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:
                print("✅ Face detected!")
                return current_frame.copy()
        time.sleep(0.1)

    print("⚠ Face detection timeout.")
    return None

def detect_nervousness():
    global current_frame
    if current_frame is None:
        return "No camera feed available"
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "⚠ No face detected"
    elif len(faces) > 1:
        return "⚠ Multiple faces detected"
    else:
        return "✅ Face detected"

# ------------------ AUDIO INPUT ------------------
def record_answer():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Calibrating mic...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("🗣 Speak your answer now (10 seconds max)...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())
            text = recognizer.recognize_google(audio)
            pitch = extract_pitch("temp.wav")
            os.remove("temp.wav")
            print("✅ Recognized:", text)
            return text, pitch
        except Exception as e:
            print(f"❌ Error recording: {str(e)}")
            return "Error", 0

# ------------------ AUDIO FEATURE EXTRACTION ------------------
def extract_pitch(filename):
    try:
        y, sr_rate = librosa.load(filename)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        return np.mean(pitch_values) if len(pitch_values) > 0 else 0
    except:
        return 0

# ------------------ EVALUATION ------------------
def evaluate_answer(question, answer):
    keywords = {
        "yourself": ["name", "student", "experience", "background"],
        "hire": ["skills", "team", "fit", "qualified"],
        "strengths": ["hardworking", "punctual", "dedicated"],
        "weaknesses": ["improve", "working on", "overcome"],
        "5 years": ["career", "goals", "position"]
    }
    score = 0
    for key in keywords:
        if key in question.lower():
            for word in keywords[key]:
                if word in answer.lower():
                    score += 1
    return f"{score}/5"

def ai_feedback(question, answer):
    prompt = f"I asked: {question}\nThe candidate answered: {answer}\nGive constructive feedback."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error from OpenAI API: {str(e)}"

# ------------------ LOGGING ------------------
def log_response(question, answer, score, feedback, nervousness, pitch):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append([timestamp, question, answer, score, feedback, nervousness, round(pitch, 2)])

def save_log():
    headers = ["Timestamp", "Question", "Answer", "Score", "Feedback", "Nervousness", "Voice Pitch"]
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(log_data)
    print(f"📝 Responses saved to {output_file}")

# ------------------ MAIN LOGIC ------------------
def run_interview():
    if initialize_camera():
        camera_thread = threading.Thread(target=camera_monitor, daemon=True)
        camera_thread.start()
        try:
            for i, question in enumerate(questions):
                print(f"\n🎓 Question {i+1}: {question}")
                frame = wait_for_face()
                if frame is None:
                    print("Skipping question due to missing face.")
                    continue

                nervousness = detect_nervousness()
                print("🧠 Nervousness Status:", nervousness)

                answer, pitch = record_answer()
                score = evaluate_answer(question, answer)
                feedback = ai_feedback(question, answer)

                print(f"🔍 Score: {score}")
                print("🤖 GPT Feedback:\n", feedback)
                print(f"🎵 Voice Pitch: {pitch:.2f} Hz")

                log_response(question, answer, score, feedback, nervousness, pitch)
                input("\n➡ Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n⚠ Interview interrupted by user.")
        finally:
            camera_running = False
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            save_log()
            print("✅ Interview finished.")
    else:
        print("❌ Camera initialization failed.")

if __name__ == "__main__":
    run_interview()
