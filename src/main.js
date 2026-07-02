import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import Stats from 'three/addons/libs/stats.module.js';
import {
  uniform, storage, positionWorld, sin, mix, float, vec3, smoothstep,
  Fn, instanceIndex, vertexIndex, If, mx_noise_vec3
} from 'three/tsl';

const PARTICLE_COUNT = 120000;   // GPU常駐パーティクル（コンピュートで更新）
const MODEL_URL = `/sample.glb`;

const overlay  = document.getElementById('overlay');
const loadmsg  = document.getElementById('loadmsg');
const phaseEl  = document.getElementById('phase');

// 失敗時は画面に理由を表示（無言のまま止まらないように）
window.addEventListener('error', e => showError(e.error || e.message));
window.addEventListener('unhandledrejection', e => showError(e.reason));

// ---------------------------------------------------------------------------
// レンダラ / シーン / カメラ
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x05060a, 1);
document.getElementById('app').appendChild(renderer.domElement);

// FPS / パフォーマンス計測パネル（クリックで FPS→MS→MB 切替）
const stats = new Stats();
stats.dom.style.cssText = 'position:fixed;top:8px;left:8px;z-index:30;';
document.body.appendChild(stats.dom);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 9);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

// 背景に淡い星屑（雰囲気づけ）
addBackgroundStars();

// ライト（標準マテリアルのメッシュは光源が無いと真っ黒になる）
scene.add(new THREE.HemisphereLight(0xbcd0ff, 0x0a0c14, 1.3));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(4, 8, 5);
scene.add(keyLight);

// ---------------------------------------------------------------------------
// 共有 uniform
// ---------------------------------------------------------------------------
const uTime          = uniform(0);
const uDelta         = uniform(0);
const uMouse         = uniform(new THREE.Vector3(9999, 9999, 9999)); // マウスの3D接触点（ワールド）
const uMouseStrength = uniform(0);                                   // 触れている強さ（0..1）
const uReveal        = uniform(0);                                   // メッシュ表示量

// ---------------------------------------------------------------------------
// GLB 読み込み
// ---------------------------------------------------------------------------
const draco = new DRACOLoader().setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.171.0/examples/jsm/libs/draco/');
const loader = new GLTFLoader();
loader.setDRACOLoader(draco);
loader.setMeshoptDecoder(MeshoptDecoder);

let particles;                 // { points, update, reset }
let modelGroup = null;         // レイキャスト対象の実メッシュ
const modelMats = [];

// マウス（ポインタ）
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
let pointerInside = false;

try {
  await renderer.init();
  const gltf = await loader.loadAsync(MODEL_URL);
  const model = gltf.scene;
  model.updateMatrixWorld(true);

  loadmsg.textContent = '表面をサンプリング中…';
  await frame();

  const sampled = sampleModel(model, PARTICLE_COUNT);
  particles = buildParticles(sampled);
  scene.add(particles.points);

  // 実メッシュをパーティクルと同じ座標系に合わせて土台として配置
  const group = new THREE.Group();
  group.scale.setScalar(sampled.scale);
  group.position.copy(sampled.center).multiplyScalar(-sampled.scale);
  group.add(model);

  // メッシュ不透明度：全体表示量(uReveal) × マウス接触点まわりのディゾルブ穴
  const DISSOLVE = 1.1;
  const nz = sin(positionWorld.x.mul(26))
    .add(sin(positionWorld.y.mul(26)))
    .add(sin(positionWorld.z.mul(26))).mul(0.06);
  const dist = positionWorld.sub(uMouse).length().add(nz);
  const hole = smoothstep(DISSOLVE * 0.35, DISSOLVE, dist);          // 0=溶ける 1=残る
  const dissolveOpacity = uReveal.mul(mix(float(1.0), hole, uMouseStrength));

  model.traverse(o => {
    if (!o.isMesh) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const mm of mats) {
      mm.transparent = true;
      mm.depthWrite = true;
      mm.opacityNode = dissolveOpacity;   // マウスでディゾルブ
      modelMats.push(mm);
    }
  });
  scene.add(group);
  modelGroup = group;

  controls.target.set(0, 0, 0);
  controls.update();

  uReveal.value = 1;              // メッシュは常時表示
  phaseEl.textContent = 'WebGPU Compute · なぞってモデルを崩す';

  // 「リセット」ボタン：剥離した粒を全て元に戻す
  const replayBtn = document.getElementById('replay');
  replayBtn.textContent = '↺ リセット';
  replayBtn.addEventListener('click', () => particles?.reset());

  overlay.style.opacity = '0';
  setTimeout(() => overlay.remove(), 650);

  animate();
} catch (err) {
  showError(err);
}

// ---------------------------------------------------------------------------
// パーティクル系（WebGPU コンピュートシェーダ）
//   各粒が 位置/速度/寿命 の状態を GPU に常駐で持ち、毎フレーム更新。
//   マウスで触れた粒は「剥離」し、カールノイズの流れ場＋浮力で漂い続ける。
// ---------------------------------------------------------------------------
function buildParticles({ positions, colors, count }) {
  // --- 状態バッファ（GPU常駐ストレージ） ---
  const posAttr  = new THREE.StorageInstancedBufferAttribute(positions.slice(), 3); // 現在位置
  const velAttr  = new THREE.StorageInstancedBufferAttribute(new Float32Array(count * 3), 3);
  const homeAttr = new THREE.StorageInstancedBufferAttribute(positions, 3);         // 元の定位置
  const colAttr  = new THREE.StorageInstancedBufferAttribute(colors, 3);
  const ageAttr  = new THREE.StorageInstancedBufferAttribute(new Float32Array(count), 1); // 0=付着, >0=剥離後の経過秒

  const posBuf  = storage(posAttr,  'vec3',  count);
  const velBuf  = storage(velAttr,  'vec3',  count);
  const homeBuf = storage(homeAttr, 'vec3',  count);
  const colBuf  = storage(colAttr,  'vec3',  count);
  const ageBuf  = storage(ageAttr,  'float', count);

  // --- 物理パラメータ ---
  const REACH = .4;   // 剥離する半径（メッシュのディゾルブと同程度）
  const FLOW  = 2.0;   // カールノイズ流れ場の強さ
  const BUOY  = 1.5;   // 浮力（上昇）
  const DRAG  = 0.7;   // 空気抵抗
  const LIFE  = 2.0;   // 剥離後の寿命（秒）→ フェード後に静かに再利用

  // カールノイズ（発散ゼロの渦流。ノイズ場を微分して curl を作る）
  const curl = Fn(([p]) => {
    const e = 0.4;
    const dx = vec3(e, 0, 0), dy = vec3(0, e, 0), dz = vec3(0, 0, e);
    const x = mx_noise_vec3(p.add(dy)).z.sub(mx_noise_vec3(p.sub(dy)).z)
      .sub(mx_noise_vec3(p.add(dz)).y.sub(mx_noise_vec3(p.sub(dz)).y));
    const y = mx_noise_vec3(p.add(dz)).x.sub(mx_noise_vec3(p.sub(dz)).x)
      .sub(mx_noise_vec3(p.add(dx)).z.sub(mx_noise_vec3(p.sub(dx)).z));
    const z = mx_noise_vec3(p.add(dx)).y.sub(mx_noise_vec3(p.sub(dx)).y)
      .sub(mx_noise_vec3(p.add(dy)).x.sub(mx_noise_vec3(p.sub(dy)).x));
    return vec3(x, y, z).mul(1.0 / (2.0 * e));
  });

  // --- 更新コンピュート（毎フレーム全粒） ---
  const updateFn = Fn(() => {
    const pos  = posBuf.element(instanceIndex);
    const vel  = velBuf.element(instanceIndex);
    const home = homeBuf.element(instanceIndex);
    const age  = ageBuf.element(instanceIndex);

    // まだ付いている粒：マウス接触点の近くなら剥離
    If(age.equal(0.0), () => {
      If(uMouseStrength.greaterThan(0.5).and(home.distance(uMouse).lessThan(REACH)), () => {
        age.assign(float(0.0001));   // 剥離マーク
        vel.assign(home.sub(uMouse).normalize().mul(0.6).add(vec3(0.0, 1.2, 0.0))); // 初速：外向き＋上
      });
    });

    // 剥離済み：流れ場＋浮力で漂う（戻らない）
    If(age.greaterThan(0.0), () => {
      const sp = pos.mul(0.7).add(vec3(0.0, uTime.mul(0.4), 0.0)); // 時間で流れ場をスクロール
      const flow = curl(sp).mul(FLOW);
      vel.addAssign(flow.add(vec3(0.0, BUOY, 0.0)).mul(uDelta));
      vel.mulAssign(float(1.0).sub(uDelta.mul(DRAG)));             // 抵抗
      pos.addAssign(vel.mul(uDelta));
      age.addAssign(uDelta);
      // 寿命後は静かに再利用（十分遠くでフェード済み）
      If(age.greaterThan(LIFE), () => {
        age.assign(float(0.0));
        pos.assign(home);
        vel.assign(vec3(0.0));
      });
    });
  })().compute(count);

  // --- リセットコンピュート（全粒を元に戻す） ---
  const resetFn = Fn(() => {
    ageBuf.element(instanceIndex).assign(float(0.0));
    posBuf.element(instanceIndex).assign(homeBuf.element(instanceIndex));
    velBuf.element(instanceIndex).assign(vec3(0.0));
  })().compute(count);

  // --- 描画（付着中は size 0 で不可視、剥離した粒だけ発光して見える） ---
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3)); // 描画数ぶんのダミー

  const material = new THREE.PointsNodeMaterial({ transparent: true, depthWrite: false });
  material.blending = THREE.AdditiveBlending;
  // Points は非インスタンス描画なので、頂点ごとに vertexIndex で明示的に読む
  material.positionNode = posBuf.element(vertexIndex);

  const age = ageBuf.element(vertexIndex);
  const fadeIn  = smoothstep(0.0, 0.3, age);          // 剥離直後にフワッと出現（付着中=0で不可視）
  const fadeOut = smoothstep(LIFE, LIFE - 1.5, age);  // 寿命末で消える
  material.sizeNode  = fadeIn.mul(fadeOut).mul(3.4);
  material.colorNode = colBuf.element(vertexIndex).mul(1.4); // モデルの色のまま発光

  const points = new THREE.Points(geo, material);
  points.frustumCulled = false;

  return {
    points,
    update: () => renderer.compute(updateFn),
    reset:  () => renderer.compute(resetFn),
  };
}

// ---------------------------------------------------------------------------
// メインループ
// ---------------------------------------------------------------------------
function animate() {
  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.033, (now - last) / 1000); last = now;
    uTime.value  = now / 1000;
    uDelta.value = dt;
    updateMouse();
    particles.update();          // GPU コンピュートで粒を更新
    controls.update();
    renderer.render(scene, camera);
    stats.update();
  });
}

// ---------------------------------------------------------------------------
// モデル表面のサンプリング（面積加重 + テクスチャ/マテリアル色）
// ---------------------------------------------------------------------------
function sampleModel(root, count) {
  const meshes = [];
  root.traverse(o => { if (o.isMesh && o.geometry?.attributes.position) meshes.push(o); });
  if (!meshes.length) throw new Error('メッシュが見つかりませんでした。');

  const prepped = meshes.map(prepMesh);
  const totalArea = prepped.reduce((s, p) => s + p.area, 0) || 1;

  const positions = new Float32Array(count * 3);
  const colors    = new Float32Array(count * 3);
  const normals   = new Float32Array(count * 3);

  const box = new THREE.Box3();
  const p = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const col = new THREE.Color();

  let written = 0;
  prepped.forEach((m, mi) => {
    const n = (mi === prepped.length - 1)
      ? count - written
      : Math.round(count * m.area / totalArea);
    for (let i = 0; i < n && written < count; i++, written++) {
      const t = pickTriangle(m.cum, m.area);
      sampleTriangle(m, t, p, col, nrm);
      positions[written * 3]     = p.x;
      positions[written * 3 + 1] = p.y;
      positions[written * 3 + 2] = p.z;
      colors[written * 3]     = col.r;
      colors[written * 3 + 1] = col.g;
      colors[written * 3 + 2] = col.b;
      normals[written * 3]     = nrm.x;
      normals[written * 3 + 1] = nrm.y;
      normals[written * 3 + 2] = nrm.z;
      box.expandByPoint(p);
    }
  });

  // 中心を原点に、最大寸法を約5ユニットへ正規化
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const scale = 5 / (Math.max(size.x, size.y, size.z) || 1);
  const SHELL = -0.03; // 実メッシュのわずか内側に沈めて z-fighting を防ぐ（負=内向き）
  for (let i = 0; i < count; i++) {
    positions[i * 3]     = (positions[i * 3]     - center.x) * scale + normals[i * 3]     * SHELL;
    positions[i * 3 + 1] = (positions[i * 3 + 1] - center.y) * scale + normals[i * 3 + 1] * SHELL;
    positions[i * 3 + 2] = (positions[i * 3 + 2] - center.z) * scale + normals[i * 3 + 2] * SHELL;
  }

  return { positions, colors, normals, count, center, scale };
}

function prepMesh(mesh) {
  const g = mesh.geometry;
  if (!g.attributes.normal) g.computeVertexNormals(); // 法線が無ければ生成
  const posAttr = g.attributes.position;
  const nAttr   = g.attributes.normal;
  const uvAttr  = g.attributes.uv || null;
  const idx     = g.index;

  // ワールド座標の頂点配列
  const wp = new Float32Array(posAttr.count * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < posAttr.count; i++) {
    v.fromBufferAttribute(posAttr, i).applyMatrix4(mesh.matrixWorld);
    wp[i * 3] = v.x; wp[i * 3 + 1] = v.y; wp[i * 3 + 2] = v.z;
  }

  // ワールド空間の頂点法線（法線行列で変換）
  const wn = new Float32Array(posAttr.count * 3);
  const nm = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
  const vn = new THREE.Vector3();
  for (let i = 0; i < posAttr.count; i++) {
    vn.fromBufferAttribute(nAttr, i).applyMatrix3(nm).normalize();
    wn[i * 3] = vn.x; wn[i * 3 + 1] = vn.y; wn[i * 3 + 2] = vn.z;
  }

  const triCount = idx ? idx.count / 3 : posAttr.count / 3;
  const gi = (t, k) => idx ? idx.getX(t * 3 + k) : t * 3 + k;

  // 三角形ごとの面積累積（面積加重サンプリング用）
  const cum = new Float32Array(triCount);
  let area = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3();
  for (let t = 0; t < triCount; t++) {
    const i0 = gi(t, 0), i1 = gi(t, 1), i2 = gi(t, 2);
    a.set(wp[i0 * 3], wp[i0 * 3 + 1], wp[i0 * 3 + 2]);
    b.set(wp[i1 * 3], wp[i1 * 3 + 1], wp[i1 * 3 + 2]);
    c.set(wp[i2 * 3], wp[i2 * 3 + 1], wp[i2 * 3 + 2]);
    ab.subVectors(b, a); ac.subVectors(c, a);
    area += 0.5 * ab.cross(ac).length();
    cum[t] = area;
  }

  // マテリアル（配列なら先頭）とベースカラー・テクスチャ
  const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const baseColor = new THREE.Color(1, 1, 1);
  if (mat?.color) baseColor.copy(mat.color); // three内部でリニア
  const tex = readTexture(mat?.map);

  return { wp, wn, uvAttr, idx, triCount, gi, cum, area, baseColor, tex };
}

// テクスチャ画像を canvas に描いてピクセル配列を取得（sRGB）
const _texCache = new Map();
function readTexture(map) {
  if (!map || !map.image) return null;
  if (_texCache.has(map)) return _texCache.get(map);
  let data = null;
  try {
    const img = map.image;
    const w = img.width, h = img.height;
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    data = { d: ctx.getImageData(0, 0, w, h).data, w, h };
  } catch (e) {
    data = null; // KTX2 等の圧縮テクスチャは読めない → マテリアル色にフォールバック
  }
  _texCache.set(map, data);
  return data;
}

// rand*area を cum 上で二分探索して三角形を選ぶ
function pickTriangle(cum, area) {
  const r = Math.random() * area;
  let lo = 0, hi = cum.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] < r) lo = mid + 1; else hi = mid;
  }
  return lo;
}

const _srgb = new THREE.Color();
function sampleTriangle(m, t, outPos, outCol, outNrm) {
  const i0 = m.gi(t, 0), i1 = m.gi(t, 1), i2 = m.gi(t, 2);
  // 重心座標（一様サンプリング）
  let u = Math.random(), w = Math.random();
  if (u + w > 1) { u = 1 - u; w = 1 - w; }
  const wp = m.wp;
  const ax = wp[i0 * 3], ay = wp[i0 * 3 + 1], az = wp[i0 * 3 + 2];
  const bx = wp[i1 * 3], by = wp[i1 * 3 + 1], bz = wp[i1 * 3 + 2];
  const cx = wp[i2 * 3], cy = wp[i2 * 3 + 1], cz = wp[i2 * 3 + 2];
  outPos.set(
    ax + u * (bx - ax) + w * (cx - ax),
    ay + u * (by - ay) + w * (cy - ay),
    az + u * (bz - az) + w * (cz - az),
  );

  // 法線を重心補間（ワールド空間）
  const wn = m.wn;
  outNrm.set(
    wn[i0 * 3]     + u * (wn[i1 * 3]     - wn[i0 * 3])     + w * (wn[i2 * 3]     - wn[i0 * 3]),
    wn[i0 * 3 + 1] + u * (wn[i1 * 3 + 1] - wn[i0 * 3 + 1]) + w * (wn[i2 * 3 + 1] - wn[i0 * 3 + 1]),
    wn[i0 * 3 + 2] + u * (wn[i1 * 3 + 2] - wn[i0 * 3 + 2]) + w * (wn[i2 * 3 + 2] - wn[i0 * 3 + 2]),
  ).normalize();

  // 色：テクスチャ優先、無ければマテリアル色
  if (m.tex && m.uvAttr) {
    const uvA = m.uvAttr;
    const uu = uvA.getX(i0) + u * (uvA.getX(i1) - uvA.getX(i0)) + w * (uvA.getX(i2) - uvA.getX(i0));
    const vv = uvA.getY(i0) + u * (uvA.getY(i1) - uvA.getY(i0)) + w * (uvA.getY(i2) - uvA.getY(i0));
    const fx = uu - Math.floor(uu);            // repeat ラップ
    const fy = vv - Math.floor(vv);
    const px = Math.min(m.tex.w - 1, Math.max(0, (fx * m.tex.w) | 0));
    const py = Math.min(m.tex.h - 1, Math.max(0, (fy * m.tex.h) | 0));
    const idx = (py * m.tex.w + px) * 4;
    _srgb.setRGB(m.tex.d[idx] / 255, m.tex.d[idx + 1] / 255, m.tex.d[idx + 2] / 255, THREE.SRGBColorSpace);
    outCol.copy(_srgb).multiply(m.baseColor); // baseColorFactor
  } else {
    outCol.copy(m.baseColor);
  }
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------
function addBackgroundStars() {
  const n = 1200;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 30 + Math.random() * 30;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
    pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
    pos[i * 3 + 2] = r * Math.cos(ph);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.PointsNodeMaterial({ color: 0x3a4a6a, size: 0.06, transparent: true, opacity: 0.6 });
  scene.add(new THREE.Points(g, m));
}

function frame() { return new Promise(r => requestAnimationFrame(r)); }

// ---------------------------------------------------------------------------
// マウス反応（レイキャストで接触点を求め、uniform を更新）
// ---------------------------------------------------------------------------
window.addEventListener('pointermove', e => {
  pointerNdc.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointerNdc.y = -(e.clientY / window.innerHeight) * 2 + 1;
  pointerInside = true;
});
window.addEventListener('pointerleave', () => { pointerInside = false; });

function updateMouse() {
  let hit = false;
  if (pointerInside && modelGroup) {
    raycaster.setFromCamera(pointerNdc, camera);
    const hits = raycaster.intersectObject(modelGroup, true);
    if (hits.length) { uMouse.value.copy(hits[0].point); hit = true; }
  }
  // 触れている強さを滑らかに増減
  const targetStrength = hit ? 1 : 0;
  uMouseStrength.value += (targetStrength - uMouseStrength.value) * 0.2;
}

function showError(err) {
  console.error(err);
  const gpu = 'gpu' in navigator;
  if (!overlay.isConnected) return;
  overlay.style.opacity = '1';
  overlay.innerHTML =
    `<div id="err"><b>読み込みに失敗しました</b><br>${err?.message || err}` +
    (gpu ? '' : '<br><br>このブラウザは WebGPU に対応していません（Chrome/Edge 113+ を推奨）。') +
    '</div>';
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
