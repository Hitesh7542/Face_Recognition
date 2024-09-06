import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from deepface import DeepFace
import tensorflow as tf
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

def check_file_permissions(file_path):
    if not os.path.exists(file_path):
        st.error(f"File {file_path} does not exist.")
        return False
    if not os.access(file_path, os.R_OK):
        st.error(f"File {file_path} is not readable.")
        return False
    if not os.access(file_path, os.W_OK):
        st.error(f"File {file_path} is not writable.")
        return False
    st.success(f"File {file_path} is accessible and has appropriate permissions.")
    return True

model = tf.keras.models.load_model('face_similar_model.h5')


required_files = ['trainer.yml', 'haarcascade_frontalface_default.xml']
for file in required_files:
    if not check_file_permissions(file):
        st.stop()

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Lenka@7542",
    database="MotherDairy"
)
cursor = conn.cursor()


cursor.execute("SELECT ConcessionerName, BoothNumber, ConcessionID, StartDate, EndDate FROM Concessioners")
metadata = {}
for (name, booth_no, concession_id, start_date, end_date) in cursor.fetchall():
    metadata[name] = {
        'booth_no': booth_no,
        'concession_name': name,
        'concession_id': concession_id,
        'active_period': f"{start_date} - {end_date}"
    }

cursor.close()
conn.close()


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX


names = ['None', 'Hitesh', 'Ram', 'Jeny', 'Sam', 'Jiban', 'Keny', 'sp', 'John', 'Sharma', 'Steve', 'Suresh', 'Kumar', 'Naveen', 'Vineet', 'Pravin', 'Asmin', 'Dipak','Happy']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_embedding(model, image):
    return model.predict(image)[0]

def load_dataset_embeddings(dataset_path, model):
    embeddings = []
    image_paths = []
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img_array = preprocess_image(img_path)
                embedding = get_embedding(model, img_array)
                embeddings.append(embedding)
                image_paths.append(img_path)
    return np.array(embeddings), image_paths

def find_similar_faces(input_image_path, dataset_dir, model, top_n=5):
    input_image = preprocess_image(input_image_path)
    input_embedding = get_embedding(model, input_image)
    results = []
    
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    dataset_image = preprocess_image(img_path)
                    dataset_embedding = get_embedding(model, dataset_image)
                    similarity = cosine_similarity([input_embedding], [dataset_embedding])[0][0]
                    name = person
                    image_metadata = metadata.get(name, {
                        'booth_no': 'N/A',
                        'concession_name': name,
                        'concession_id': 'N/A',
                        'active_period': 'N/A'
                    })
                    results.append((img_path, similarity, image_metadata))
                except Exception as e:
                    st.write(f"Error processing {img_path}: {e}")
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

def save_uploaded_file(uploaded_file):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def detect_and_recognize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    
    metadata_list = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
        if confidence < 100:
            name = names[id_]
            metadata_info = {
                'Name': name,
                'Booth No': metadata.get(name, {}).get('booth_no', 'N/A'),
                'Concession Name': metadata.get(name, {}).get('concession_name', 'N/A'),
                'Concession ID': metadata.get(name, {}).get('concession_id', 'N/A'),
                'Active Period': metadata.get(name, {}).get('active_period', 'N/A')
            }
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "unknown"
            metadata_info = {
                'Name': name,
                'Booth No': 'N/A',
                'Concession Name': 'N/A',
                'Concession ID': 'N/A',
                'Active Period': 'N/A'
            }
            confidence_text = f"  {round(100 - confidence)}%"
        
        cv2.putText(image, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(image, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
        metadata_list.append(metadata_info)
    
    return image, metadata_list

st.markdown("""
    <style>
    .main {
        background-color: #262424;
        font-family: Arial, sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        background-color: #e7e7e7;
        border-radius: 12px;
        padding: 10px;
    }
    .stTextInput input {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 8px;
        font-size: 16px;
    }
    .image-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        margin-top: 20px;
    }
    .image-grid img {
        margin: 10px;
        border: 2px solid #4CAF50;
        border-radius: 12px;
        max-width: 200px;
        transition: transform 0.2s;
    }
    .image-grid img:hover {
        transform: scale(1.05);
        cursor: pointer;
    }
    .metadata-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    .metadata-table th, .metadata-table td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
        font-size: 14px;
    }
    .metadata-table th {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Face Recognition Web App')
st.markdown("<h2 style='color: #4CAF50;'>Upload an image</h2>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
dataset_path = r"C:\Users\hites\Desktop\Code_Data\Task\Face\Similarity\dataset\train"
folder_path = r"C:\Users\hites\Desktop\Code_Data\Task\Face\FacialRecognition\Dataset"

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    st.write("Press 'c' to capture the image and 'q' to quit.")
    
    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            captured_image = frame
            break
        elif key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_image

if st.button('Capture Image from Webcam'):
    captured_image = capture_image_from_webcam()
    if captured_image is not None:
        processed_image, metadata_list = detect_and_recognize(captured_image)
        st.image(processed_image, caption='Captured Image', use_column_width=True)
        if metadata_list:
            st.subheader('Metadata')
            metadata_df = pd.DataFrame(metadata_list)
            st.table(metadata_df.style.set_table_attributes("class='metadata-table'"))

if st.button('Identify Image'):
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            image_np = np.array(image.convert('RGB'))
            
            processed_image, metadata_list = detect_and_recognize(image_np)
            
            st.image(image, caption='Given Image', use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(processed_image, caption='Processed Image', use_column_width=True)
            with col2:
                if metadata_list:
                    st.subheader('Metadata')
                    metadata_df = pd.DataFrame(metadata_list)
                    st.table(metadata_df.style.set_table_attributes("class='metadata-table'"))
        except Exception as e:
            st.write(f"Error: {e}")
    else:
        st.write("Please upload an image.")

if st.button('Display Images'):
    if folder_path:
        try:
            images = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('jpg', 'jpeg', 'png')):
                        image_path = os.path.join(root, file)
                        images.append(image_path)
            
            st.write(f'Total {len(images)} images found.')

            if images:
                st.markdown("<div class='image-grid'>", unsafe_allow_html=True)
                for img_path in images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=os.path.basename(img_path), use_column_width=False)
                    except Exception as e:
                        st.write(f"Error loading image: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.write(f"Error accessing folder: {e}")
    else:
        st.write("Please enter a valid folder path.")

if st.button('Find Similar Images'):
    if uploaded_image is not None and dataset_path:
        try:
            temp_image_path = 'temp_image.jpg'
            with open(temp_image_path, 'wb') as f:
                f.write(uploaded_image.getvalue())
            
            similar_images = find_similar_faces(temp_image_path, dataset_path, model)
            
            if similar_images:
                st.write(f"Found {len(similar_images)} similar images:")
                for img_path, similarity, metadata_info in similar_images:
                    try:
                        col1, col2 = st.columns(2)
                        with col1:
                            image = Image.open(img_path)
                            st.image(image, caption=f"{os.path.basename(img_path)}\nSimilarity: {similarity * 100:.2f}%", use_column_width=True)
                        with col2:
                            metadata_df = pd.DataFrame([metadata_info])
                            st.table(metadata_df.style.set_table_attributes("class='metadata-table'"))
                    except Exception as e:
                        st.write(f"Error loading image: {e}")
            else:
                st.write("No similar images found.")
        
        except Exception as e:
            st.write(f"Error finding similar images: {e}")
    else:
        st.write("Please upload an image and ensure the dataset path is valid.")
