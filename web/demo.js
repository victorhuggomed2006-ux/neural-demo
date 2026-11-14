/* Interactive demo: draw a digit, prepare it like MNIST, run the network, show the result. */
(function () {
  'use strict';
  var canvas = document.getElementById('nn-canvas');
  if (!canvas || !window.Neural || !window.NEURAL_WEIGHTS) return;
  var $ = function (s) { return document.getElementById(s); };
  var ctx = canvas.getContext('2d', { willReadFrequently: true });
  var net = window.Neural.load(window.NEURAL_WEIGHTS);
  var bigEl = $('nn-digit'), barsEl = $('nn-bars'), seen = $('nn-seen'), seenCtx = seen.getContext('2d');
  var mapsEl = $('nn-maps'), hintEl = $('nn-hint');
  var drawing = false, last = null, timer = 0, dirty = false;

  /* 10 bars */
  var bars = [];
  for (var d = 0; d < 10; d++) {
    var li = document.createElement('li');
    li.innerHTML = '<span>' + d + '</span><i><b></b></i><em>0%</em>';
    barsEl.appendChild(li); bars.push(li);
  }
  /* feature-map canvases: 8 after the first block, 16 after the second */
  var mapCanvases = [];
  [[8, 14, 'nn-maps-1'], [16, 7, 'nn-maps-2']].forEach(function (g) {
    var row = $(g[2]); var list = [];
    for (var i = 0; i < g[0]; i++) {
      var c = document.createElement('canvas'); c.width = g[1]; c.height = g[1]; c.setAttribute('aria-hidden', 'true');
      row.appendChild(c); list.push(c);
    }
    mapCanvases.push(list);
  });

  function setupCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 20; ctx.strokeStyle = '#ffffff';
  }

  /* MNIST-style preparation: crop the stroke, fit it in 20x20, centre it by its centre of mass in 28x28 */
  function prepare() {
    var w = canvas.width, h = canvas.height, data = ctx.getImageData(0, 0, w, h).data;
    var minX = w, minY = h, maxX = -1, maxY = -1;
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 40) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
    }
    if (maxX < 0) return null;
    var bw = maxX - minX + 1, bh = maxY - minY + 1, scale = 20 / Math.max(bw, bh), dw = bw * scale, dh = bh * scale;
    var a = document.createElement('canvas'); a.width = a.height = 28; var ac = a.getContext('2d', { willReadFrequently: true });
    ac.fillStyle = '#000'; ac.fillRect(0, 0, 28, 28); ac.imageSmoothingEnabled = true; ac.imageSmoothingQuality = 'high';
    ac.drawImage(canvas, minX, minY, bw, bh, (28 - dw) / 2, (28 - dh) / 2, dw, dh);
    var px = ac.getImageData(0, 0, 28, 28).data, sum = 0, sx = 0, sy = 0;
    for (var i = 0; i < 28; i++) for (var j = 0; j < 28; j++) { var v = px[(i * 28 + j) * 4]; sum += v; sx += v * j; sy += v * i; }
    var shiftX = Math.round(14 - sx / sum), shiftY = Math.round(14 - sy / sum);
    var b = document.createElement('canvas'); b.width = b.height = 28; var bc = b.getContext('2d', { willReadFrequently: true });
    bc.fillStyle = '#000'; bc.fillRect(0, 0, 28, 28); bc.drawImage(a, shiftX, shiftY);
    var out = bc.getImageData(0, 0, 28, 28).data, pixels = new Float32Array(784);
    for (var k = 0; k < 784; k++) pixels[k] = out[k * 4] / 255;
    return pixels;
  }

  function drawSeen(pixels) {
    var img = seenCtx.createImageData(28, 28);
    for (var i = 0; i < 784; i++) { var v = Math.round(pixels[i] * 255); img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255; }
    seenCtx.putImageData(img, 0, 0);
  }
  function drawMaps(maps) {
    maps.forEach(function (m, gi) {
      var list = mapCanvases[gi], size = m.h;
      for (var k = 0; k < m.c; k++) {
        var cx = list[k].getContext('2d'), img = cx.createImageData(size, size), base = k * size * size, mx = 1e-6, i;
        for (i = 0; i < size * size; i++) if (m.data[base + i] > mx) mx = m.data[base + i];
        for (i = 0; i < size * size; i++) {
          var v = Math.round(255 * m.data[base + i] / mx);
          img.data[i * 4] = 229; img.data[i * 4 + 1] = 84; img.data[i * 4 + 2] = 11; img.data[i * 4 + 3] = v;
        }
        cx.clearRect(0, 0, size, size); cx.putImageData(img, 0, 0);
      }
    });
  }
  function showResult(res) {
    bigEl.textContent = res.digit;
    bars.forEach(function (li, d) {
      var p = res.probs[d];
      li.classList.toggle('best', d === res.digit);
      li.querySelector('b').style.width = (p * 100).toFixed(1) + '%';
      li.querySelector('em').textContent = (p * 100 < 0.5 ? '<1' : Math.round(p * 100)) + '%';
    });
    drawMaps(res.acts.maps);
  }
  function clearResult() {
    bigEl.textContent = '–';
    bars.forEach(function (li) { li.classList.remove('best'); li.querySelector('b').style.width = '0'; li.querySelector('em').textContent = '0%'; });
    seenCtx.fillStyle = '#000'; seenCtx.fillRect(0, 0, 28, 28);
    mapCanvases.forEach(function (list) { list.forEach(function (c) { c.getContext('2d').clearRect(0, 0, c.width, c.height); }); });
    if (hintEl) hintEl.hidden = false;
  }
  function run() {
    var pixels = prepare();
    if (!pixels) { clearResult(); return; }
    if (hintEl) hintEl.hidden = true;
    drawSeen(pixels);
    showResult(net.predict(pixels));
    dirty = false;
  }
  function schedule() {
    dirty = true;
    if (timer) return;
    timer = setTimeout(function () { timer = 0; if (dirty) run(); }, 70);
  }

  function pos(e) {
    var r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height };
  }
  canvas.addEventListener('pointerdown', function (e) {
    e.preventDefault(); canvas.setPointerCapture(e.pointerId);
    drawing = true; last = pos(e);
    ctx.beginPath(); ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, 6.2832); ctx.fillStyle = '#fff'; ctx.fill();
    if (hintEl) hintEl.hidden = true;
    schedule();
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!drawing) return;
    var p = pos(e);
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke(); last = p;
    schedule();
  });
  function end() { if (drawing) { drawing = false; run(); } }
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  $('nn-clear').addEventListener('click', function () { setupCanvas(); clearResult(); });

  /* test hook: lets automated checks draw strokes (does nothing for visitors) */
  window.__nn = { run: run, draw: function (fn) { fn(ctx); run(); }, clear: function () { setupCanvas(); clearResult(); } };

  setupCanvas(); clearResult();
})();
