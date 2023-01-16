// Datos de simulacion fijos
const carreras = [
  'Computación',
  'Electricidad',
  'Telemática',
  'Electrónica y Automatización',
  'Telecomunicaciones'
];

const horarios = [
  '7:00 - 9:00',
  '9:00 - 11:00',
  '11:00 - 13:00',
  '14:00 - 16:00',
  '16:00 - 18:00'
];

const cursos = [
  'Inteligecia Artificial',
  'Ingenieria de Software',
  'Redes de datos',
  'Sistemas Operativos'
];

// Registros de la base de datos
const data = JSON.parse(registers).registers;

// Funciones de analisis de datos
const numeroPorCarrera = (carrera) => {
  let num = 0;
  for (let register of data) {
    if (register.carrera === carrera) {
      num = num + 1;
    }
  }
  return num;
};

const numeroPorCurso = (curso) => {
  let num = 0;
  for (let register of data) {
    if (register.curso === curso) {
      num = num + 1;
    }
  }
  return num;
};

const numeroPorHorario = (horario) => {
  let num = 0;
  for (let register of data) {
    if (register.horario === horario) {
      num = num + 1;
    }
  }
  return num;
};

const getMaxRegisterCurso = () => {
  const numPorCurso = cursos.map((c) => numeroPorCurso(c));
  return cursos[numPorCurso.indexOf(Math.max(...numPorCurso))];
};

const getMaxRegisterCarrera = () => {
  const numPorCarrera = carreras.map((c) => numeroPorCarrera(c));
  return carreras[numPorCarrera.indexOf(Math.max(...numPorCarrera))];
};

const getMaxRegisterHorario = () => {
  const numPorHorario = horarios.map((h) => numeroPorHorario(h));
  return horarios[numPorHorario.indexOf(Math.max(...numPorHorario))];
};

const getFechas = () => {
  const fechas = [];
  for (let register of data) {
    fechas.push(register.fecha);
  }
  return fechas;
};

// Bar chart
const barChart = document.getElementById('bar_chart');
const numPorCurso = cursos.map((c) => numeroPorCurso(c));
const dataBar = {
  labels: cursos,
  datasets: [
    {
      label: 'Número de registros',
      data: numPorCurso,
      borderWidth: 1
    }
  ]
};
new Chart(barChart, {
  type: 'bar',
  data: dataBar,
  options: {
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
});

// Line chart
const numPorHorario = horarios.map((h) => numeroPorHorario(h));
const lineChart = document.getElementById('line_chart');
const dataLine = {
  labels: horarios,
  datasets: [
    {
      label: 'Número de registros',
      data: numPorHorario,
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0
    }
  ]
};
new Chart(lineChart, {
  type: 'line',
  data: dataLine
});

// Polar Area Chart
const polar_chart = document.getElementById('polar_chart');
const numPorCarrera = carreras.map((c) => numeroPorCarrera(c));
const dataPolar = {
  labels: carreras,
  datasets: [
    {
      label: 'Número de registros',
      data: numPorCarrera,
      backgroundColor: [
        'rgb(255, 99, 132)',
        'rgb(75, 192, 192)',
        'rgb(255, 205, 86)',
        'rgb(201, 203, 207)',
        'rgb(54, 162, 235)'
      ]
    }
  ]
};
new Chart(polar_chart, {
  type: 'polarArea',
  data: dataPolar
});

// Resume section
const totalRegsiters = document.getElementById('total_registers');
totalRegsiters.textContent = data.length;

const maxRegistersCurso = document.getElementById('max_registers_curso');
maxRegistersCurso.textContent = getMaxRegisterCurso();

const maxRegistersCarrera = document.getElementById('max_registers_carrera');
maxRegistersCarrera.textContent = getMaxRegisterCarrera();

const maxRegistersHorario = document.getElementById('max_registers_horario');
maxRegistersHorario.textContent = getMaxRegisterHorario();
