from flask import Flask, render_template, Response, request
from flask_mysqldb import MySQL
import yaml
from functions import registers, generate

# App creation
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = b'_1#y2l"F4Q8z\n\xec]/'

# Configure db
db = yaml.load(open('db/db.yaml'), Loader=yaml.Loader)
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

variable = 0
videoStart = False

registers = []


def register_tuple(dic):
    register_list = []
    for key, value in dic.items():
        if key == 'paralelo':
            register_list.append(int(value))
        else:
            register_list.append(value)

    return tuple(register_list)


def save_to_db():
    global registers
    print(registers)
    cur = mysql.connection.cursor()
    query = "INSERT INTO registers(fecha, curso, paralelo, horario, carrera, facultad) VALUES(%s, %s, %s, %s, %s, %s)"
    for r in registers:
        cur.execute(query, register_tuple(r))
    mysql.connection.commit()
    cur.close()


def register_dict(arr):
    return dict(id=arr[0],
                hora=arr[1],
                fecha=arr[2],
                curso=arr[3],
                paralelo=arr[4],
                horario=arr[5],
                carrera=arr[6],
                facultad=arr[7],
                )


def get_db_registers():
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM registers")
    db_registers = []

    if resultValue > 0:
        db_registers = cur.fetchall()

    db_registers = list(db_registers)
    db_registers_formatted = []

    for register in db_registers:
        r = list(register)
        current_time = r[1]
        hora = f"{current_time.hour}:{current_time.minute}:{current_time.second}"
        fecha = f"{current_time.day}/{current_time.month}/{current_time.year}"
        r.insert(2, fecha)  # Solo fecha
        r[1] = hora  # Solo hora
        db_registers_formatted.append(register_dict(r))

    return db_registers_formatted


@app.route('/video')
def video():
    global registers
    return Response(generate(registers), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def index():
    global variable
    global videoStart
    global registers
    videoStart = False
    if request.method == "POST":
        variable = int(request.form.get("variable"))
        videoStart = True if variable == 1 else False
        if videoStart:
            registers = []
            print('video start')
        else:
            print('video terminado')
            save_to_db()
    return render_template('clase.html', videoStart=videoStart)


@app.route('/registros')
def registros():
    db_registers = get_db_registers()
    return render_template('registros.html', registers=db_registers)


@app.route('/graficos')
def graficos():
    db_registers = get_db_registers()
    json_registers = {'registers': db_registers}
    return render_template('graficos.html', registers=json_registers)


if __name__ == "__main__":
    app.run(debug=True)
