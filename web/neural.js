/* Forward pass of the digit CNN, written by hand (no libraries).
 * Weights come from neural-demo (PyTorch), exported as float16 + base64 in window.NEURAL_WEIGHTS.
 * Architecture: conv3x3 -> relu -> pool2 -> conv3x3 -> relu -> pool2 -> flatten -> dense -> relu -> dense. */
(function (root) {
  'use strict';

  function halfToFloat(h) {
    var s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function decode(b64) {
    var bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
    for (var i = 0; i < n; i++) out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
    return out;
  }

  function conv3x3(x, cin, h, w, weight, bias, cout) {
    var y = new Float32Array(cout * h * w);
    for (var o = 0; o < cout; o++) {
      for (var yy = 0; yy < h; yy++) {
        for (var xx = 0; xx < w; xx++) {
          var sum = bias[o];
          for (var c = 0; c < cin; c++) {
            var base = (o * cin + c) * 9, plane = c * h * w;
            for (var ky = 0; ky < 3; ky++) {
              var iy = yy + ky - 1;
              if (iy < 0 || iy >= h) continue;
              for (var kx = 0; kx < 3; kx++) {
                var ix = xx + kx - 1;
                if (ix < 0 || ix >= w) continue;
                sum += weight[base + ky * 3 + kx] * x[plane + iy * w + ix];
              }
            }
          }
          y[(o * h + yy) * w + xx] = sum;
        }
      }
    }
    return y;
  }
  function relu(x) { for (var i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }
  function pool2(x, c, h, w) {
    var oh = h >> 1, ow = w >> 1, y = new Float32Array(c * oh * ow);
    for (var k = 0; k < c; k++)
      for (var i = 0; i < oh; i++)
        for (var j = 0; j < ow; j++) {
          var b = k * h * w + 2 * i * w + 2 * j;
          y[(k * oh + i) * ow + j] = Math.max(x[b], x[b + 1], x[b + w], x[b + w + 1]);
        }
    return y;
  }
  function dense(x, weight, bias, out) {
    var n = x.length, y = new Float32Array(out);
    for (var o = 0; o < out; o++) {
      var sum = bias[o], base = o * n;
      for (var i = 0; i < n; i++) sum += weight[base + i] * x[i];
      y[o] = sum;
    }
    return y;
  }
  function softmax(z) {
    var m = -Infinity, i, s = 0, p = new Float32Array(z.length);
    for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    for (i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
    for (i = 0; i < z.length; i++) p[i] /= s;
    return p;
  }

  function load(payload) {
    var layers = payload.layers.map(function (l) {
      return { kind: l.kind, shape: l.shape, w: decode(l.w), b: decode(l.b) };
    });
    var ops = payload.architecture;

    /* pixels: Float32Array(784), values in [0, 1], row-major 28x28 */
    function predict(pixels) {
      var x = pixels, c = 1, h = 28, w = 28, li = 0, acts = { input: pixels, maps: [] };
      for (var k = 0; k < ops.length; k++) {
        var op = ops[k];
        if (op === 'conv') {
          var L = layers[li++];
          x = conv3x3(x, c, h, w, L.w, L.b, L.shape[0]);
          c = L.shape[0];
        } else if (op === 'relu') {
          x = relu(x);
        } else if (op === 'pool') {
          x = pool2(x, c, h, w); h >>= 1; w >>= 1;
          acts.maps.push({ data: x, c: c, h: h, w: w });
        } else if (op === 'flatten') {
          c = x.length; h = 1; w = 1;
        } else if (op === 'dense') {
          var D = layers[li++];
          x = dense(x, D.w, D.b, D.shape[0]);
          acts.hidden = acts.hidden || x;
        }
      }
      var probs = softmax(x), best = 0;
      for (var i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      return { logits: x, probs: probs, digit: best, acts: acts, accuracy: payload.test_accuracy };
    }
    return { predict: predict, layers: layers, accuracy: payload.test_accuracy };
  }

  var api = { load: load, decode: decode };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Neural = api;
})(typeof window !== 'undefined' ? window : globalThis);
