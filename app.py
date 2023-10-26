from flask import Flask, render_template, Response
import cv2
from PIL import Image
from PIL import ImageTk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

app = Flask(__name__)

face_model = Sequential()

# Add Conv2D layers, MaxPooling2D layers, and other layers here to match the architecture of recognition_model
face_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
face_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Dropout(0.25))


# Load the saved weights
face_model.load_weights('recognition_model.h5')

# Dictionary for facial expressions
facial_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Global variables
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

# Function to capture image and recognize emotion
def Capture_Image():
    global cap1
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("Cannot open the camera")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))
    # Detect faces in the video and draw bounding boxes
    bounding_box = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_frame = gray_frame[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
        prediction = face_model.predict(crop_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, facial_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex

    if flag1 is None:
        print("Error!")

    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, Capture_Image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def Get_Emoji():
    frame2 = cv2.imread(f'emojis/{show_text[0]}.png')
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=facial_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(10, Get_Emoji)

def generate():
    while True:
        Capture_Image()
        frame1 = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame1)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
