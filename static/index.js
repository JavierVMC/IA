const tableBody = document.getElementById('table_body');

for (let i = 0; i < 20; i++) {
  tableBody.innerHTML += `
  <tr>
    <td>1</td>
    <td>11:45</td>
    <td>20/12/2022</td>
    <td>Inteligencia Artificial</td>
    <td>2</td>
    <td>11:00 - 13:00</td>
    <td>Computacion</td>
    <td>FIEC</td>
  </tr>
  `;
}
