const $ = id => document.getElementById(id);
const video = $('video'), cap = document.createElement('canvas'), out = $('out');
let stream = null, running = false, busy = false, uploaded = null, sel = {};
const ids = () => Object.values(sel).map(p => p.id);
const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(r => r.json());

function toggle(id) {
  const p = PRODUCTS.find(x => x.id == id); if (!p) return;
  if (sel[p.category] && sel[p.category].id === p.id) delete sel[p.category];
  else {
    sel[p.category] = p;
    if (p.category === 'dress') { delete sel.top; delete sel.bottom; }
    if (p.category === 'top' || p.category === 'bottom') delete sel.dress;
  }
  render();
}
function render() {
  document.querySelectorAll('.pick').forEach(e => e.classList.toggle('active', ids().includes(+e.dataset.id)));
  $('chosen').innerHTML = Object.values(sel).map(p => `<span class="badge bg-dark me-1">${p.name}</span>`).join('') || '<span class="text-muted">Nothing selected</span>';
  $('total').textContent = Object.values(sel).reduce((s, p) => s + p.price, 0);
  if (uploaded) sendFrame();
}
async function sendFrame() {
  let img, mirror;
  if (uploaded) { img = uploaded; mirror = false; }
  else if (running && video.videoWidth) {
    cap.width = 512; cap.height = Math.round(512 * video.videoHeight / video.videoWidth);
    cap.getContext('2d').drawImage(video, 0, 0, cap.width, cap.height);
    img = cap.toDataURL('image/jpeg', 0.7); mirror = true;
  } else return;
  if (busy) return; busy = true;
  try {
    const d = await post('/api/tryon', {image: img, ids: ids(), scale: +$('scale').value, dy: +$('dy').value, height: +$('height').value || 170, mirror});
    if (d.image) { out.src = d.image; out.classList.remove('d-none'); $('ph').classList.add('d-none'); }
    $('status').textContent = d.detected ? 'Body detected ✔' : 'No full body detected – step back so shoulders and hips are visible.';
    if (d.measurements) { $('shoulder').textContent = d.measurements.shoulder_cm; $('poseSize').textContent = d.measurements.size; }
  } catch (e) { $('status').textContent = 'Server error – is the Flask app running?'; }
  busy = false;
}
async function loop() { if (!running) return; await sendFrame(); setTimeout(loop, 40); }

$('start').onclick = async () => {
  try {
    uploaded = null;
    stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
    video.srcObject = stream; await video.play(); running = true; loop();
  } catch (e) { $('status').textContent = 'Camera error: ' + e.message + ' (use http://127.0.0.1:5000 and allow camera access)'; }
};
$('stop').onclick = () => { running = false; if (stream) stream.getTracks().forEach(t => t.stop()); };
$('file').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  $('stop').click(); const r = new FileReader();
  r.onload = () => { uploaded = r.result; sendFrame(); }; r.readAsDataURL(f);
};
$('shot').onclick = () => {
  if (out.classList.contains('d-none')) return alert('Start the camera first.');
  const a = document.createElement('a'); a.href = out.src; a.download = 'tryon_' + Date.now() + '.jpg'; a.click();
};
$('save').onclick = async () => {
  if (!ids().length) return alert('Select clothes first.');
  const d = await post('/outfit/save', {ids: ids(), name: prompt('Outfit name?', 'My outfit') || 'My outfit'}); alert(d.msg);
};
$('cartbtn').onclick = async () => {
  if (!ids().length) return alert('Select clothes first.');
  const d = await post('/outfit/cart', {ids: ids()}); alert(d.msg);
};
$('sizebtn').onclick = async () => {
  const d = await post('/api/size', {height: $('height').value, weight: $('weight').value, gender: $('gender').value});
  $('mlSize').textContent = d.size + ' (' + d.confidence + '%)';
};
['scale', 'dy'].forEach(i => $(i).addEventListener('input', () => { $('sv').textContent = $('scale').value; if (uploaded) sendFrame(); }));
document.querySelectorAll('.pick').forEach(e => e.onclick = () => toggle(e.dataset.id));
INITIAL.forEach(toggle); render();

document.querySelectorAll('.cat-tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.cat-tab').forEach(x => { x.classList.toggle('btn-dark', x === b); x.classList.toggle('btn-outline-dark', x !== b); });
  document.querySelectorAll('.pickcol').forEach(c => c.style.display = (b.dataset.cat === 'all' || c.dataset.cat === b.dataset.cat) ? '' : 'none');
});
