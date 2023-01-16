import datetime
import random
import cv2

import numpy as np
import tensorflow as tf

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
registers = []

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


def saveToRegisters(registers):
    now = datetime.datetime.now()
    formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
    data = dict(
        fecha=formatted_date,
        curso=cursos[random.randint(0, len(cursos) - 1)],
        paralelo=random.randint(1, 9),
        horario=horarios[random.randint(0, len(horarios) - 1)],
        carrera=carreras[random.randint(0, len(carreras) - 1)],
        facultad='FIEC')
    registers.append(data)
    print(registers)


def generate(registers):
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
                    eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)
                    if classify(eye_roi) == 1:
                        cerrados = cerrados + 1
                        saveToRegisters(registers)
                    else:
                        abiertos = abiertos + 1
                print(f'Abiertos: {abiertos}')
                print(f'Cerrados: {cerrados}')

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
