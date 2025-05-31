from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import pandas as pd
import time
from flask_socketio import SocketIO, emit
import base64
import json
import sqlite3
import uuid
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Drop existing table if it exists
    c.execute('DROP TABLE IF EXISTS users')
    
    # Create new table with all required columns
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id TEXT PRIMARY KEY, 
                  name TEXT, 
                  email TEXT, 
                  face_encoding TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Global variables
known_face_encodings = []
known_face_names = []
attendance_list = []
last_attendance_time = {}
ATTENDANCE_INTERVAL = 300  # 5 minutes in seconds

# Load known faces from database
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, name, face_encoding FROM users')
    rows = c.fetchall()
    
    for row in rows:
        face_encoding = np.array(json.loads(row[2]))
        known_face_encodings.append(face_encoding)
        known_face_names.append(row[0])  # Using ID as the identifier
    
    conn.close()

# Initialize known faces
load_known_faces()

# Load attendance data
def load_attendance():
    global attendance_list
    if os.path.exists('Attendance.csv'):
        df = pd.read_csv('Attendance.csv')
        attendance_list = df.to_dict('records')
    else:
        attendance_list = []

# Save attendance data
def save_attendance():
    df = pd.DataFrame(attendance_list)
    df.to_csv('Attendance.csv', index=False)

# Initialize attendance data
load_attendance()

# Ensure the uploads directory exists
os.makedirs('uploads', exist_ok=True)

# Function to save login details to an Excel sheet
def save_login_details(name, image_path):
    df = pd.DataFrame({'Name': [name], 'Image Path': [image_path]})
    df.to_excel('login_details.xlsx', index=False, mode='a', header=not os.path.exists('login_details.xlsx'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/verify/single')
def verify_single():
    return render_template('verify_single.html')

@app.route('/verify/multiple')
def verify_multiple():
    return render_template('verify_multiple.html')

@app.route('/attendance')
def attendance():
    period = request.args.get('period', 'all')
    date = request.args.get('date')
    month = request.args.get('month')
    year = request.args.get('year')
    # Filter out records with NaN values
    filtered_attendance = [record for record in attendance_list if str(record.get('name')).lower() != 'nan' and str(record.get('id')).lower() != 'nan' and str(record.get('time')).lower() != 'nan']
    def parse_time(record):
        try:
            return datetime.strptime(record['time'], '%Y-%m-%d %H:%M:%S')
        except Exception:
            return None
    if period == 'daily' and date:
        filtered_attendance = [r for r in filtered_attendance if parse_time(r) and parse_time(r).strftime('%Y-%m-%d') == date]
    elif period == 'monthly' and month:
        filtered_attendance = [r for r in filtered_attendance if parse_time(r) and parse_time(r).strftime('%Y-%m') == month]
    elif period == 'yearly' and year:
        filtered_attendance = [r for r in filtered_attendance if parse_time(r) and parse_time(r).strftime('%Y') == year]
    # For monthly/yearly, organize by date
    grouped_attendance = None
    if period in ['monthly', 'yearly']:
        grouped_attendance = {}
        for r in filtered_attendance:
            d = parse_time(r).strftime('%Y-%m-%d')
            if d not in grouped_attendance:
                grouped_attendance[d] = []
            grouped_attendance[d].append(r)
    else:
        grouped_attendance = None
    return render_template('attendance.html', attendance=filtered_attendance, period=period, date=date, month=month, year=year, grouped_attendance=grouped_attendance)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        # Process the uploaded image for face recognition
        image = face_recognition.load_image_file(filename)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            # Assume the first face is the user
            face_encoding = face_encodings[0]
            # Compare with known faces and save login details
            # (This is a placeholder; you should implement actual face comparison logic)
            save_login_details('User', filename)
            return jsonify({'success': 'File uploaded and processed'}), 200
        else:
            return jsonify({'error': 'No face detected'}), 400

@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            # Assume the first face is the user
            face_encoding = face_encodings[0]
            # Save the image to uploads directory
            filename = f"uploads/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            # Save login details
            save_login_details('User', filename)
            return jsonify({'success': 'Live image captured and processed'}), 200
        else:
            return jsonify({'error': 'No face detected'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        user_id = data.get('id')
        image_data = data.get('image')
        
        if not all([name, email, user_id, image_data]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if not face_encodings:
            return jsonify({'success': False, 'error': 'No face detected in the image'})
        
        if len(face_encodings) > 1:
            return jsonify({'success': False, 'error': 'Multiple faces detected. Please provide an image with only one face'})
        
        face_encoding = face_encodings[0]
        
        # Save image
        image_filename = f"{user_id}_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join('static', 'images', 'faces', image_filename)
        cv2.imwrite(image_path, image)
        
        # Save to database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO users (id, name, email, face_encoding, image_path) VALUES (?, ?, ?, ?, ?)',
                 (user_id, name, email, json.dumps(face_encoding.tolist()), image_path))
        conn.commit()
        conn.close()
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/verify/single', methods=['POST'])
def api_verify_single():
    try:
        data = request.json
        user_id = data.get('id')
        image_data = data.get('image')
        
        if not all([user_id, image_data]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save verification image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"verify_{user_id}_{timestamp}.jpg"
        image_path = os.path.join('static', 'images', 'verification', image_filename)
        cv2.imwrite(image_path, image)
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if not face_encodings:
            return jsonify({'success': False, 'error': 'No face detected in the image'})
        
        if len(face_encodings) > 1:
            return jsonify({'success': False, 'error': 'Multiple faces detected. Please use multiple face verification'})
        
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            matched_id = known_face_names[first_match_index]
            
            if matched_id == user_id:
                # Get user details
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('SELECT name FROM users WHERE id = ?', (user_id,))
                user = c.fetchone()
                conn.close()
                
                if user:
                    # Record attendance
                    attendance_list.append({
                        'name': user[0],
                        'id': user_id,
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'image_path': image_path
                    })
                    save_attendance()
                    socketio.emit('attendance_update', {'name': user[0], 'id': user_id, 'time': attendance_list[-1]['time']})
                    
                    return jsonify({'success': True, 'name': user[0]})
        
        return jsonify({'success': False, 'error': 'Face verification failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/verify/multiple', methods=['POST'])
def api_verify_multiple():
    try:
        data = request.json
        user_ids = data.get('ids', [])
        image_data = data.get('image')
        
        if not user_ids or not image_data:
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save verification image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"verify_multiple_{timestamp}.jpg"
        image_path = os.path.join('static', 'images', 'verification', image_filename)
        cv2.imwrite(image_path, image)
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if not face_encodings:
            return jsonify({'success': False, 'error': 'No faces detected in the image'})
        
        if len(face_encodings) < len(user_ids):
            return jsonify({'success': False, 'error': f'Not enough faces detected. Expected {len(user_ids)}, found {len(face_encodings)}'})
        
        if len(face_encodings) > len(user_ids):
            return jsonify({'success': False, 'error': f'Too many faces detected. Expected {len(user_ids)}, found {len(face_encodings)}'})
        
        results = []
        verified_ids = set()
        
        # Compare each face with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            if True in matches:
                first_match_index = matches.index(True)
                matched_id = known_face_names[first_match_index]
                
                if matched_id in user_ids and matched_id not in verified_ids:
                    # Get user details
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute('SELECT name FROM users WHERE id = ?', (matched_id,))
                    user = c.fetchone()
                    conn.close()
                    
                    if user:
                        # Record attendance
                        attendance_list.append({
                            'name': user[0],
                            'id': matched_id,
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'image_path': image_path
                        })
                        save_attendance()
                        socketio.emit('attendance_update', {'name': user[0], 'id': matched_id, 'time': attendance_list[-1]['time']})
                        
                        results.append({
                            'id': matched_id,
                            'name': user[0],
                            'success': True
                        })
                        verified_ids.add(matched_id)
        
        # Check if all IDs were verified
        unverified_ids = set(user_ids) - verified_ids
        if unverified_ids:
            return jsonify({
                'success': False,
                'error': f'Could not verify IDs: {", ".join(unverified_ids)}',
                'results': results
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_frame(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
            # Check if enough time has passed since last attendance
            current_time = time.time()
            if name not in last_attendance_time or (current_time - last_attendance_time[name]) > ATTENDANCE_INTERVAL:
                last_attendance_time[name] = current_time
                
                # Get user details
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('SELECT name FROM users WHERE id = ?', (name,))
                user = c.fetchone()
                conn.close()
                
                if user:
                    # Save frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    image_filename = f"auto_{name}_{timestamp}.jpg"
                    image_path = os.path.join('static', 'images', 'verification', image_filename)
                    cv2.imwrite(image_path, frame)
                    
                    attendance_list.append({
                        'name': user[0],
                        'id': name,
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'image_path': image_path
                    })
                    save_attendance()
                    socketio.emit('attendance_update', {'name': user[0], 'id': name, 'time': attendance_list[-1]['time']})
        
        face_names.append(name)
    
    # Draw boxes and names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
    
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('attendance_update', {'attendance': attendance_list})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/images/faces', exist_ok=True)
    os.makedirs('static/images/verification', exist_ok=True)
    socketio.run(app, debug=True)
