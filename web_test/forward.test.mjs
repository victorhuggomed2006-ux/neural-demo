// Checks that the hand-written JavaScript forward pass reproduces the Python reference.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const Neural = require(path.join(here, '..', 'web', 'neural.js'));

const sandbox = {};
new Function('window', readFileSync(path.join(here, '..', 'artifacts', 'weights.js'), 'utf8'))(sandbox);
const net = Neural.load(sandbox.NEURAL_WEIGHTS);
const ref = JSON.parse(readFileSync(path.join(here, '..', 'artifacts', 'reference.json'), 'utf8'));

let worst = 0, wrong = 0;
for (const [i, s] of ref.samples.entries()) {
  const px = Float32Array.from(s.pixels, (v) => v / 255);
  const out = net.predict(px);
  const diff = Math.max(...s.logits.map((v, k) => Math.abs(v - out.logits[k])));
  worst = Math.max(worst, diff);
  if (out.digit !== s.digit) { wrong++; console.error(`sample ${i}: predicted ${out.digit}, expected ${s.digit}`); }
}
console.log(`samples=${ref.samples.length} worst logit difference=${worst.toExponential(2)} wrong digits=${wrong}`);
if (worst > 1e-3 || wrong > 0) { console.error('FAIL'); process.exit(1); }
console.log('OK: JavaScript matches the Python reference');
