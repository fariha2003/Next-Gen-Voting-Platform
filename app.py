from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pickle
import matplotlib.pyplot as plt
import io
import base64
import time
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define paths
DATA_DIR = "data"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uid = request.form['uid']
        name = request.form['name']
        if not uid or not name:
            return "UID and Name are required", 400

        # Capture face
        face_data_path = os.path.join(DATA_DIR, uid)
        if not os.path.exists(face_data_path):
            os.makedirs(face_data_path)

        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(os.path.join(face_data_path, f"{count}.jpg"), face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capturing Faces", frame)
            if cv2.waitKey(1) == 27 or count >= 100:  # Escape key or 10 samples
                break
        cap.release()
        cv2.destroyAllWindows()

        # Upload fingerprint
        fingerprint = request.files['fingerprint']
        fingerprint_path = os.path.join(DATA_DIR, f"{uid}_fingerprint.tif")
        fingerprint.save(fingerprint_path)

        # Save user data
        user_data_file = os.path.join(DATA_DIR, "users.pkl")
        users = {}
        if os.path.exists(user_data_file):
            with open(user_data_file, "rb") as f:
                users = pickle.load(f)
        
        users[uid] = {"name": name, "fingerprint": fingerprint_path}
        with open(user_data_file, "wb") as f:
            pickle.dump(users, f)

        return redirect(url_for('home'))

    return render_template('register.html')

@app.route('/vote', methods=['GET', 'POST'])
def vote():
    if request.method == 'POST':
        uid = request.form['uid']
        name = request.form['name']
        fingerprint = request.files['fingerprint']
        candidate = request.form['candidate']

        # Verify UID and name
        user_data_file = os.path.join(DATA_DIR, "users.pkl")
        with open(user_data_file, "rb") as f:
            users = pickle.load(f)

        if uid not in users or users[uid]["name"] != name:
            return "Invalid UID or Name", 400

        # Verify face and fingerprint
        if not verify_face(uid) or not verify_fingerprint(uid, fingerprint):
            return "Face or fingerprint mismatch", 400

        # Record vote
        candidates_file = os.path.join(DATA_DIR, "candidates.pkl")
        with open(candidates_file, "rb") as f:
            candidates = pickle.load(f)

        candidates[candidate] += 1
        with open(candidates_file, "wb") as f:
            pickle.dump(candidates, f)

        return redirect(url_for('home'))

    # Fetch candidates for voting
    candidates_file = os.path.join(DATA_DIR, "candidates.pkl")
    with open(candidates_file, "rb") as f:
        candidates = pickle.load(f)

    return render_template('vote.html', candidates=candidates)

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/add_candidate', methods=['POST'])
def add_candidate():
    candidate_name = request.form['candidate_name']
    candidates_file = os.path.join(DATA_DIR, "candidates.pkl")
    
    if os.path.exists(candidates_file):
        with open(candidates_file, "rb") as f:
            candidates = pickle.load(f)
    else:
        candidates = {}

    candidates[candidate_name] = 0
    with open(candidates_file, "wb") as f:
        pickle.dump(candidates, f)

    return redirect(url_for('admin'))

@app.route('/view_results')
def view_results():
    candidates_file = os.path.join(DATA_DIR, "candidates.pkl")
    with open(candidates_file, "rb") as f:
        candidates = pickle.load(f)

    # Generate pie chart
    labels = candidates.keys()
    sizes = candidates.values()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    
    # Convert plot to PNG
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)

    return render_template('results.html', img_base64=img_base64)


def verify_face(uid):
    face_data_path = os.path.join(DATA_DIR, uid)
    if not os.path.exists(face_data_path):
        return False

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while time.time() - start_time <= 45:  # Limit to 45 seconds
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            captured_face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            for filename in os.listdir(face_data_path):
                registered_face = cv2.imread(os.path.join(face_data_path, filename), cv2.IMREAD_GRAYSCALE)
                score, _ = ssim(captured_face, registered_face, full=True)
                if score > 0.75:
                    cap.release()
                    cv2.destroyAllWindows()
                    return True

    cap.release()
    cv2.destroyAllWindows()
    return False


def verify_fingerprint(uid, fingerprint):
    user_data_file = os.path.join(DATA_DIR, "users.pkl")
    with open(user_data_file, "rb") as f:
        users = pickle.load(f)
    
    registered_fingerprint = users[uid]["fingerprint"]
    try:
        registered_image = Image.open(registered_fingerprint).convert("L")
        fingerprint_image = Image.open(fingerprint).convert("L")
        registered_image = np.array(registered_image)
        fingerprint_image = np.array(fingerprint_image)
        score, _ = ssim(registered_image, fingerprint_image, full=True)
        return score > 0.75
    except:
        return False

if __name__ == '__main__':
    app.run(debug=True)
