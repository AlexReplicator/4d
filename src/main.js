/* 4D FPS Slice - WebGL2 raymarcher */
(() => {
'use strict';
const canvas = document.getElementById('glcanvas');
const dpr = Math.min(2, window.devicePixelRatio || 1);
let renderScale = 1.5;
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance', desynchronized: true });
if (!gl) { alert('WebGL2 не поддерживается в этом браузере'); return; }

function resize() {
  const w = Math.floor(window.innerWidth * dpr * renderScale);
  const h = Math.floor(window.innerHeight * dpr * renderScale);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

const VERT_SRC = `#version 300 es
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
out vec4 fragColor;
in vec2 v_uv;
uniform vec2 u_res;
uniform float u_time;
uniform vec3 u_camPos;
uniform mat3 u_camRot;
uniform float u_w;
uniform vec3 u_rot4;
uniform int u_frame;
uniform float u_eps;
uniform int u_maxSteps;

float sdSphere4(vec4 p, float r){ return length(p) - r; }
float sdBox4(vec4 p, vec4 b){
  vec4 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(max(q.x, q.y), max(q.z, q.w)), 0.0);
}

vec4 rotXW(vec4 p, float a){ float c=cos(a), s=sin(a); return vec4(c*p.x - s*p.w, p.y, p.z, s*p.x + c*p.w); }
vec4 rotYW(vec4 p, float a){ float c=cos(a), s=sin(a); return vec4(p.x, c*p.y - s*p.w, p.z, s*p.y + c*p.w); }
vec4 rotZW(vec4 p, float a){ float c=cos(a), s=sin(a); return vec4(p.x, p.y, c*p.z - s*p.w, s*p.z + c*p.w); }

float smin(float a, float b, float k){
  float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0 - h);
}

float map3(vec3 p){
  vec4 q = vec4(p, u_w);
  q = rotXW(q, u_rot4.x);
  q = rotYW(q, u_rot4.y);
  q = rotZW(q, u_rot4.z);
  float d1 = sdSphere4(q, 1.3);
  float d2 = sdBox4(q - vec4(1.6, 0.0, 0.0, 0.0), vec4(0.7));
  float d3 = sdBox4(q + vec4(0.0, 1.2, 0.0, 0.0), vec4(0.5, 0.9, 0.5, 0.5));
  float d = smin(d1, d2, 0.25);
  d = smin(d, d3, 0.25);
  return d;
}

vec3 calcNormal(vec3 p){
  float e = max(u_eps * 1.5, 0.0007);
  vec2 h = vec2(e, 0.0);
  float dx = map3(p + vec3(h.x, h.y, h.y)) - map3(p - vec3(h.x, h.y, h.y));
  float dy = map3(p + vec3(h.y, h.x, h.y)) - map3(p - vec3(h.y, h.x, h.y));
  float dz = map3(p + vec3(h.y, h.y, h.x)) - map3(p - vec3(h.y, h.y, h.x));
  return normalize(vec3(dx, dy, dz));
}

float raymarch(in vec3 ro, in vec3 rd, out vec3 outPos, out int steps){
  const float MAX_DIST = 100.0;
  float SURF_DIST = u_eps;
  const int MAX_STEPS_CAP = 256;
  float t = 0.0;
  steps = 0;
  for(int i=0;i<MAX_STEPS_CAP;i++){
    if(i >= u_maxSteps){ steps = i; break; }
    vec3 p = ro + rd * t;
    float d = map3(p);
    if(d < SURF_DIST){ outPos = p; steps = i; return t; }
    t += d * 0.9;
    if(t > MAX_DIST){ steps = i; break; }
  }
  outPos = ro + rd * t;
  return -1.0;
}

vec3 shade(vec3 p, vec3 rd, vec3 n, float t){
  vec3 lightDir = normalize(vec3(0.6, 0.8, 0.4));
  float diff = max(dot(n, lightDir), 0.0);
  float fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
  vec3 base = 0.5 + 0.5*cos(vec3(0.0, 2.0, 4.0) + (p*0.6 + vec3(u_w*1.5)).xzy);
  vec3 col = base * (0.25 + 0.75*diff) + 0.25*fres;
  float fog = 1.0 - exp(-0.04 * t);
  col = mix(col, vec3(0.03,0.05,0.07), fog);
  return pow(col, vec3(1.0/2.2));
}

void main(){
  vec2 res = u_res;
  vec2 uv = v_uv * 2.0 - 1.0;
  float aspect = res.x / res.y;
  uv.x *= aspect;
  float fov = radians(75.0);
  float tanHalf = tan(0.5 * fov);
  vec3 dirCam = normalize(vec3(uv * tanHalf, 1.0));
  vec3 rd = normalize(u_camRot * dirCam);
  vec3 ro = u_camPos;
  vec3 p; int steps;
  float t = raymarch(ro, rd, p, steps);
  vec3 col;
  if(t > 0.0){
    vec3 n = calcNormal(p);
    col = shade(p, rd, n, t);
  } else {
    float h = 0.5 + 0.5*uv.y;
    col = mix(vec3(0.02,0.04,0.07), vec3(0.12,0.14,0.18), h);
  }
  fragColor = vec4(col, 1.0);
}`;

function createShader(gl, type, src){
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(sh);
    console.error(info);
    throw new Error('Shader compile error: ' + info);
  }
  return sh;
}

function createProgram(gl, vs, fs){
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog);
    console.error(info);
    throw new Error('Program link error: ' + info);
  }
  return prog;
}

const vs = createShader(gl, gl.VERTEX_SHADER, VERT_SRC);
const fs = createShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
const prog = createProgram(gl, vs, fs);
gl.useProgram(prog);

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

const u_res = gl.getUniformLocation(prog, 'u_res');
const u_time = gl.getUniformLocation(prog, 'u_time');
const u_camPos = gl.getUniformLocation(prog, 'u_camPos');
const u_camRot = gl.getUniformLocation(prog, 'u_camRot');
const u_w = gl.getUniformLocation(prog, 'u_w');
const u_rot4 = gl.getUniformLocation(prog, 'u_rot4');
const u_frame = gl.getUniformLocation(prog, 'u_frame');
const u_eps = gl.getUniformLocation(prog, 'u_eps');
const u_maxSteps = gl.getUniformLocation(prog, 'u_maxSteps');

const hudFps = document.getElementById('fps');
const hudPos = document.getElementById('pos');
const hudAngles = document.getElementById('angles');
const hudW = document.getElementById('w');
const hudRot = document.getElementById('rot');

const elScale = document.getElementById('qscale');
const elScaleView = document.getElementById('qview');
const elSteps = document.getElementById('maxsteps');
const elStepsView = document.getElementById('stepsview');
const elEps = document.getElementById('eps');
const elEpsView = document.getElementById('epsview');

let cam = { pos: new Float32Array([0, 0, -4]), yaw: 0, pitch: 0 };
let sliceW = 0.0;
let rot4 = { xw: 0, yw: 0, zw: 0 };
let eps = 0.0005;
let maxSteps = 256;

const keys = new Set();
window.addEventListener('keydown', (e) => { if (e.repeat) return; keys.add(e.code); if (e.code === 'KeyR') { reset(); } });
window.addEventListener('keyup', (e) => { keys.delete(e.code); });

// HUD controls (качество/точность)
if (typeof elScale !== 'undefined' && elScale) {
  const applyScale = () => {
    renderScale = (parseFloat(elScale.value) || 100) / 100.0;
    if (elScaleView) elScaleView.textContent = Math.round(renderScale * 100) + '%';
    resize();
  };
  elScale.addEventListener('input', applyScale);
  applyScale();
}
if (typeof elSteps !== 'undefined' && elSteps) {
  const applySteps = () => {
    maxSteps = parseInt(elSteps.value, 10) || 160;
    if (elStepsView) elStepsView.textContent = String(maxSteps);
  };
  elSteps.addEventListener('input', applySteps);
  applySteps();
}
if (typeof elEps !== 'undefined' && elEps) {
  const applyEps = () => {
    eps = parseFloat(elEps.value) || 0.001;
    if (elEpsView) elEpsView.textContent = eps.toFixed(4);
  };
  elEps.addEventListener('input', applyEps);
  applyEps();
}

function requestLock() {
  const anyCanvas = canvas;
  const req = anyCanvas.requestPointerLock || anyCanvas.mozRequestPointerLock;
  if (document.pointerLockElement !== canvas && req) req.call(anyCanvas);
}
canvas.addEventListener('click', requestLock);

window.addEventListener('mousemove', (e) => {
  if (document.pointerLockElement === canvas) {
    const sens = 0.0025;
    cam.yaw -= e.movementX * sens;
    cam.pitch += -e.movementY * sens;
    const lim = Math.PI / 2 - 0.01;
    cam.pitch = Math.max(-lim, Math.min(lim, cam.pitch));
  }
});

function reset() {
  cam.pos.set([0, 0, -4]);
  cam.yaw = 0;
  cam.pitch = 0;
  sliceW = 0.0;
  rot4.xw = 0; rot4.yw = 0; rot4.zw = 0;
}

function yawPitchToMat3(yaw, pitch){
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const sp = Math.sin(pitch), cp = Math.cos(pitch);
  // Forward (camera looks along +Z in camera space)
  const fx = sy * cp;
  const fy = sp;
  const fz = cy * cp;
  // Keep horizon stable: Right depends only on yaw
  let rx =  cy;
  let ry =  0.0;
  let rz = -sy;
  // Up = normalize(cross(right, forward))
  let ux = ry*fz - rz*fy;
  let uy = rz*fx - rx*fz;
  let uz = rx*fy - ry*fx;
  const ulen = Math.hypot(ux, uy, uz) || 1.0;
  ux /= ulen; uy /= ulen; uz /= ulen;
  // Column-major: [right, up, forward]
  return new Float32Array([
    rx, ry, rz,
    ux, uy, uz,
    fx, fy, fz
  ]);
}

let last = performance.now();
let frameId = 0;
let smoothedFps = 0;

function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }

function step(dt) {
  const speed = (keys.has('ShiftLeft') || keys.has('ShiftRight')) ? 6.0 : 3.0;
  const rotRate = 1.2;
  let vx = 0, vy = 0, vz = 0;
  if (keys.has('KeyW')) vz += 1;
  if (keys.has('KeyS')) vz -= 1;
  if (keys.has('KeyD')) vx += 1;
  if (keys.has('KeyA')) vx -= 1;
  if (keys.has('Space')) vy += 1;
  if (keys.has('ControlLeft') || keys.has('ControlRight')) vy -= 1;
  // 4D controls
  if (keys.has('BracketLeft')) sliceW -= 0.8 * dt;
  if (keys.has('BracketRight')) sliceW += 0.8 * dt;
  if (keys.has('KeyU')) rot4.xw += rotRate * dt;
  if (keys.has('KeyJ')) rot4.xw -= rotRate * dt;
  if (keys.has('KeyI')) rot4.yw += rotRate * dt;
  if (keys.has('KeyK')) rot4.yw -= rotRate * dt;
  if (keys.has('KeyO')) rot4.zw += rotRate * dt;
  if (keys.has('KeyL')) rot4.zw -= rotRate * dt;
  sliceW = clamp(sliceW, -3.0, 3.0);
  const len = Math.hypot(vx, vy, vz) || 1;
  vx /= len; vy /= len; vz /= len;
  const R = yawPitchToMat3(cam.yaw, cam.pitch);
  const right = [R[0], R[1], R[2]];
  const up    = [R[3], R[4], R[5]];
  const fwd   = [R[6], R[7], R[8]];
  cam.pos[0] += (right[0]*vx + up[0]*vy + fwd[0]*vz) * speed * dt;
  cam.pos[1] += (right[1]*vx + up[1]*vy + fwd[1]*vz) * speed * dt;
  cam.pos[2] += (right[2]*vx + up[2]*vy + fwd[2]*vz) * speed * dt;

  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.uniform2f(u_res, canvas.width, canvas.height);
  gl.uniform1f(u_time, performance.now() * 0.001);
  gl.uniform3f(u_camPos, cam.pos[0], cam.pos[1], cam.pos[2]);
  gl.uniformMatrix3fv(u_camRot, false, R);
  gl.uniform1f(u_w, sliceW);
  gl.uniform3f(u_rot4, rot4.xw, rot4.yw, rot4.zw);
  gl.uniform1f(u_eps, eps);
  gl.uniform1i(u_maxSteps, maxSteps);
  if (u_frame) gl.uniform1i(u_frame, frameId++);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function loop(now) {
  resize();
  const dt = Math.min(0.05, Math.max(0.0001, (now - last) / 1000));
  last = now;
  step(dt);
  const fps = 1000 / Math.max(1, (performance.now() - now));
  smoothedFps = smoothedFps * 0.9 + fps * 0.1;
  if (hudFps) hudFps.textContent = (smoothedFps|0).toString();
  if (hudPos) hudPos.textContent = `${cam.pos[0].toFixed(2)},${cam.pos[1].toFixed(2)},${cam.pos[2].toFixed(2)}`;
  if (hudAngles) hudAngles.textContent = `${(cam.yaw*180/Math.PI).toFixed(0)}° / ${(cam.pitch*180/Math.PI).toFixed(0)}°`;
  if (hudW) hudW.textContent = sliceW.toFixed(2);
  if (hudRot) hudRot.textContent = `${(rot4.xw*180/Math.PI).toFixed(0)}°/${(rot4.yw*180/Math.PI).toFixed(0)}°/${(rot4.zw*180/Math.PI).toFixed(0)}°`;
  requestAnimationFrame(loop);
}

reset();
requestAnimationFrame(loop);

})();