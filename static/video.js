const videoStart = document.getElementById('videoStart');
const videoEnd = document.getElementById('videoEnd');
let showVideo = false;

videoStart.addEventListener('click', () => {
  showVideo = true;
});

videoEnd.addEventListener('click', () => {
  showVideo = false;
});
