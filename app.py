import json
import cv2
from flask import Flask, render_template, Response, request, redirect, flash, url_for
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving.saved_model import load as saved_model_load
import h5py
import datetime
import random

# Datos de simulacion
horarios = ['7:00 - 9:00', '9:00 - 11:00',
            '11:00 - 13:00', '14:00 - 16:00', '16:00 - 18:00']
carreras = ['Computación', 'Electricidad', 'Telemática',
            'Electrónica y Automatización', 'Telecomunicaciones']
cursos = ['Inteligecia Artificial', 'Ingenieria de Software',
          'Redes de datos', 'Sistemas Operativos']
# Conteo
abiertos = 0
cerrados = 0
registers = [dict(hora='12:34:23',
                  fecha='12/12/2022',
                  curso='Inteligencia Artificial',
                  paralelo=2,
                  horario='11:00 - 13:00',
                  carrera='Computación',
                  facultad='FIEC')]

# Reconocimiento de ojos
model = tf.keras.models.load_model('modelo.h5', custom_objects={
                                   'Functional': tf.keras.models.Model})

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_face(frame):
    faces = faceCascade.detectMultiScale(frame, 1.1, 7)
    return faces


def detect_eyes(roi_gray):
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return eyes


def classify(eye_img):
    img_preprocessed = cv2.resize(eye_img, (180, 180))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    prediction = model(img_preprocessed)
    return np.argmax(prediction[0])


def saveToRegisters():
    global registers
    current_time = datetime.datetime.now()
    time = f"{current_time.hour}:{current_time.minute}:{current_time.second}"
    date = f"{current_time.day}/{current_time.month}/{current_time.year}"
    data = dict(hora=time,
                fecha=date,
                curso=cursos[random.randint(0, len(cursos) - 1)],
                paralelo=random.randint(1, 9),
                horario=horarios[random.randint(0, len(horarios) - 1)],
                carrera=carreras[random.randint(0, len(carreras) - 1)],
                facultad='FIEC')
    registers.append(data)
    print(registers)


def generate():
    global abiertos
    global cerrados
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            faces = detect_face(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = detect_eyes(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)
                    if classify(eyes_roi) == 1:
                        cerrados = cerrados + 1
                        saveToRegisters()
                    else:
                        abiertos = abiertos + 1
                print(f'Abiertos: {abiertos}')
                print(f'Cerrados: {cerrados}')

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


# Creación de app

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = b'_1#y2l"F4Q8z\n\xec]/'

variable = 0
videoStart = False


@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def index():
    global variable
    global videoStart
    if request.method == "POST":
        variable = int(request.form.get("variable"))
        videoStart = True if variable == 1 else False
    return render_template('clase.html', videoStart=videoStart)


@app.route('/registros')
def registros():
    global registers
    return render_template('registros.html', registers=registers)


@app.route('/graficos')
def graficos():
    global registers
    json_registers = {'registers': registers}
    return render_template('graficos.html', registers=json_registers)


if __name__ == "__main__":
    app.run(debug=True)
