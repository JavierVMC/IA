const barChart = document.getElementById('bar_chart');

// Datos de simulacion fijos
const carreras = [
  'Computación',
  'Electricidad',
  'Telemática',
  'Electrónica y Automatización',
  'Telecomunicaciones'
];

const data = JSON.parse(registers).registers;
console.log(data);

const numeroPorCarrera = (carrera) => {
  let num = 0;
  for (let register of data) {
    if (register.carrera === carrera) {
      num = num + 1;
    }
  }
  return num;
};

const numPorCarrera = carreras.map((c) => numeroPorCarrera(c));

new Chart(barChart, {
  type: 'bar',
  data: {
    labels: carreras,
    datasets: [
      {
        label: '# de registros',
        data: numPorCarrera,
        borderWidth: 1
      }
    ]
  },
  options: {
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
});
