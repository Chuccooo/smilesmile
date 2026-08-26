'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'demo';
type Mood = 'neutral' | 'smile' | 'laugh';

type Particle = {
  x: number; y: number; px: number; py: number; vx: number; vy: number;
  radius: number; life: number; maxLife: number;
  kind: 'rain' | 'spark'; color: string;
};
type FaceCollider = { cx: number; cy: number; rx: number; ry: number; visible: boolean };
type Blendshape = { categoryName: string; score: number };
type Landmark = { x: number; y: number; z: number };
type FaceResult = {
  faceBlendshapes?: Array<{ categories: Blendshape[] }>;
  faceLandmarks?: Landmark[][];
};
type FaceLandmarkerLike = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => FaceResult;
  close?: () => void;
};

const VISION_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const COLORS = ['#ffc857', '#ff5d8f', '#a78bfa', '#5eead4', '#60a5fa', '#f8fafc'];

function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function scoreOf(categories: Blendshape[], name: string) {
  return categories.find((item) => item.categoryName === name)?.score ?? 0;
}

function friendlyCameraError(error: unknown) {
  const name = error instanceof DOMException ? error.name : '';
  if (name === 'NotAllowedError') return '摄像头权限被拒绝。请在浏览器地址栏重新允许，然后再试一次。';
  if (name === 'NotFoundError') return '没有找到可用摄像头。你仍可进入演示模式查看完整效果。';
  if (name === 'NotReadableError') return '摄像头正被其他应用占用，请关闭其他视频应用后重试。';
  if (!window.isSecureContext) return '摄像头仅能在 HTTPS 或 localhost 中使用。';
  return '摄像头或表情模型启动失败，请刷新后重试，也可以先体验演示模式。';
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarkerLike | null>(null);
  const frameRef = useRef(0);
  const resizeCleanupRef = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [mood, setMood] = useState<Mood>('neutral');
  const [error, setError] = useState('');
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState({ fps: 0, inference: 0, particles: 0, tier: '标准' });

  const stopExperience = useCallback(() => {
    cancelAnimationFrame(frameRef.current);
    resizeCleanupRef.current?.();
    resizeCleanupRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    landmarkerRef.current?.close?.();
    landmarkerRef.current = null;
  }, []);

  const runExperience = useCallback((mode: 'camera' | 'demo') => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const context = canvas.getContext('2d', { alpha: true });
    if (!context) return;

    const lowPower = (navigator.hardwareConcurrency || 8) <= 4 ||
      ((navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8) <= 4;
    const maxParticles = lowPower ? 220 : 420;
    const inferenceInterval = lowPower ? 84 : 50;
    const collider: FaceCollider = { cx: 0, cy: 0, rx: 0, ry: 0, visible: false };
    const particles: Particle[] = [];
    let width = 1;
    let height = 1;
    let lastFrame = performance.now();
    let lastInference = 0;
    let lastMetricUpdate = performance.now();
    let framesForMetric = 0;
    let inferenceCount = 0;
    let smileActive = false;
    let laughActive = false;
    let lastFirework = -3000;
    let smoothSmile = 0;
    let smoothLaugh = 0;
    let currentMood: Mood = 'neutral';

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const mapLandmark = (point: Landmark) => {
      const sourceW = video.videoWidth || 1280;
      const sourceH = video.videoHeight || 720;
      const scale = Math.max(width / sourceW, height / sourceH);
      const drawnW = sourceW * scale;
      const drawnH = sourceH * scale;
      return {
        x: (width - drawnW) / 2 + (1 - point.x) * drawnW,
        y: (height - drawnH) / 2 + point.y * drawnH,
      };
    };

    const updateCollider = (landmarks?: Landmark[]) => {
      if (!landmarks?.length) { collider.visible = false; return; }
      let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
      for (const landmark of landmarks) {
        const point = mapLandmark(landmark);
        minX = Math.min(minX, point.x); minY = Math.min(minY, point.y);
        maxX = Math.max(maxX, point.x); maxY = Math.max(maxY, point.y);
      }
      const target = {
        cx: (minX + maxX) / 2, cy: (minY + maxY) / 2,
        rx: Math.max(38, (maxX - minX) * 0.58), ry: Math.max(50, (maxY - minY) * 0.58),
      };
      const smoothing = collider.visible ? 0.28 : 1;
      collider.cx += (target.cx - collider.cx) * smoothing;
      collider.cy += (target.cy - collider.cy) * smoothing;
      collider.rx += (target.rx - collider.rx) * smoothing;
      collider.ry += (target.ry - collider.ry) * smoothing;
      collider.visible = true;
    };

    const spawnFirework = (now: number) => {
      if (now - lastFirework < 1800) return;
      lastFirework = now;
      const originX = collider.visible ? collider.cx : width * 0.5;
      const originY = collider.visible ? Math.max(100, collider.cy - collider.ry * 1.3) : height * 0.38;
      const count = lowPower ? 72 : 120;
      for (let index = 0; index < count && particles.length < maxParticles; index += 1) {
        const angle = (index / count) * Math.PI * 2 + Math.random() * 0.08;
        const speed = 95 + Math.random() * 245;
        particles.push({
          x: originX, y: originY, px: originX, py: originY,
          vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
          radius: 1.8 + Math.random() * 2.8, life: 1.25 + Math.random() * 1.1,
          maxLife: 2.35, kind: 'spark', color: COLORS[index % COLORS.length],
        });
      }
    };

    const spawnRain = (dt: number) => {
      const amount = Math.min(8, Math.ceil(dt * (lowPower ? 70 : 120)));
      for (let index = 0; index < amount && particles.length < maxParticles; index += 1) {
        const x = Math.random() * width; const y = -20 - Math.random() * 70;
        particles.push({
          x, y, px: x, py: y, vx: -18 + Math.random() * 14, vy: 420 + Math.random() * 220,
          radius: 1.3 + Math.random(), life: 2.4, maxLife: 2.4, kind: 'rain', color: '#76d7ff',
        });
      }
    };

    const collideWithHead = (particle: Particle) => {
      if (!collider.visible) return;
      const expandedRx = collider.rx + particle.radius;
      const expandedRy = collider.ry + particle.radius;
      const dx = particle.x - collider.cx; const dy = particle.y - collider.cy;
      const value = (dx * dx) / (expandedRx * expandedRx) + (dy * dy) / (expandedRy * expandedRy);
      if (value >= 1) return;
      const length = Math.sqrt(value) || 0.001;
      const nx = dx / (expandedRx * expandedRx * length);
      const ny = dy / (expandedRy * expandedRy * length);
      const normalLength = Math.hypot(nx, ny) || 1;
      const ux = nx / normalLength; const uy = ny / normalLength;
      const boundaryScale = 1 / length;
      particle.x = collider.cx + dx * boundaryScale;
      particle.y = collider.cy + dy * boundaryScale;
      const velocityAlongNormal = particle.vx * ux + particle.vy * uy;
      if (velocityAlongNormal < 0) {
        const bounce = particle.kind === 'rain' ? 0.34 : 0.72;
        particle.vx -= (1 + bounce) * velocityAlongNormal * ux;
        particle.vy -= (1 + bounce) * velocityAlongNormal * uy;
      }
      particle.vx += ux * (particle.kind === 'rain' ? 24 : 42);
    };

    const updateMood = (smile: number, jawOpen: number, now: number) => {
      smoothSmile += (smile - smoothSmile) * 0.32;
      const rawLaugh = clamp(smile * 0.72 + jawOpen * 0.62);
      smoothLaugh += (rawLaugh - smoothLaugh) * 0.34;
      if (!smileActive && smoothSmile >= 0.43) smileActive = true;
      if (smileActive && smoothSmile <= 0.3) smileActive = false;
      if (!laughActive && smoothLaugh >= 0.68 && jawOpen >= 0.28) {
        laughActive = true; spawnFirework(now);
      }
      if (laughActive && (smoothLaugh <= 0.48 || jawOpen <= 0.16)) laughActive = false;
      const nextMood: Mood = laughActive ? 'laugh' : smileActive ? 'smile' : 'neutral';
      if (nextMood !== currentMood) { currentMood = nextMood; setMood(nextMood); }
    };

    const infer = (now: number) => {
      if (mode === 'demo') {
        const cycle = (now / 1000) % 9;
        const demoSmile = cycle > 1.2 && cycle < 7.6 ? 0.68 : 0.08;
        const demoJaw = cycle > 4.2 && cycle < 5.8 ? 0.78 : 0.05;
        collider.visible = true;
        collider.cx = width * 0.5 + Math.sin(now / 900) * width * 0.16;
        collider.cy = height * 0.52 + Math.cos(now / 1100) * 16;
        collider.rx = Math.min(width, height) * 0.15; collider.ry = collider.rx * 1.25;
        updateMood(demoSmile, demoJaw, now); inferenceCount += 1; return;
      }
      if (!landmarkerRef.current || video.readyState < 2) return;
      const result = landmarkerRef.current.detectForVideo(video, now);
      const categories = result.faceBlendshapes?.[0]?.categories ?? [];
      if (!categories.length) { updateCollider(undefined); updateMood(0, 0, now); return; }
      const smile = (scoreOf(categories, 'mouthSmileLeft') + scoreOf(categories, 'mouthSmileRight')) / 2;
      updateMood(smile, scoreOf(categories, 'jawOpen'), now);
      updateCollider(result.faceLandmarks?.[0]); inferenceCount += 1;
    };

    const draw = (now: number) => {
      frameRef.current = requestAnimationFrame(draw);
      if (document.hidden) return;
      const dt = Math.min((now - lastFrame) / 1000, 0.034);
      lastFrame = now; framesForMetric += 1;
      if (now - lastInference >= inferenceInterval) {
        lastInference = now;
        try { infer(now); } catch { /* Ignore one corrupt or late camera frame. */ }
      }
      if (smileActive) spawnRain(dt);
      context.clearRect(0, 0, width, height);
      context.save(); context.globalCompositeOperation = 'lighter';
      for (let index = particles.length - 1; index >= 0; index -= 1) {
        const particle = particles[index];
        particle.px = particle.x; particle.py = particle.y; particle.life -= dt;
        particle.vy += (particle.kind === 'rain' ? 520 : 125) * dt;
        particle.vx *= Math.pow(particle.kind === 'rain' ? 0.994 : 0.985, dt * 60);
        particle.x += particle.vx * dt; particle.y += particle.vy * dt;
        collideWithHead(particle);
        if (particle.life <= 0 || particle.y > height + 80 || particle.x < -100 || particle.x > width + 100) {
          particles.splice(index, 1); continue;
        }
        const alpha = clamp(particle.life / Math.min(particle.maxLife, 0.7));
        context.globalAlpha = alpha * (particle.kind === 'rain' ? 0.72 : 0.95);
        context.strokeStyle = particle.color; context.fillStyle = particle.color;
        if (particle.kind === 'rain') {
          context.lineWidth = particle.radius; context.beginPath();
          context.moveTo(particle.px, particle.py - 7); context.lineTo(particle.x, particle.y + 8); context.stroke();
        } else {
          context.beginPath(); context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2); context.fill();
          context.globalAlpha *= 0.35; context.lineWidth = particle.radius * 0.8; context.beginPath();
          context.moveTo(particle.px, particle.py); context.lineTo(particle.x - particle.vx * 0.025, particle.y - particle.vy * 0.025); context.stroke();
        }
      }
      context.restore();
      if (mode === 'demo' && collider.visible) {
        context.save(); context.setLineDash([7, 8]); context.lineWidth = 1.5;
        context.strokeStyle = 'rgba(255,255,255,.4)'; context.beginPath();
        context.ellipse(collider.cx, collider.cy, collider.rx, collider.ry, 0, 0, Math.PI * 2); context.stroke(); context.restore();
      }
      if (now - lastMetricUpdate > 1000) {
        const elapsed = Math.max(0.1, (now - lastMetricUpdate) / 1000);
        setMetrics({ fps: Math.round(framesForMetric / elapsed), inference: Math.round(inferenceCount / elapsed), particles: particles.length, tier: lowPower ? '节能' : '标准' });
        lastMetricUpdate = now; framesForMetric = 0; inferenceCount = 0;
      }
    };

    resize(); window.addEventListener('resize', resize);
    resizeCleanupRef.current = () => window.removeEventListener('resize', resize);
    frameRef.current = requestAnimationFrame(draw);
  }, []);

  const startCamera = useCallback(async () => {
    if (status === 'loading') return;
    setStatus('loading'); setError(''); stopExperience();
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new DOMException('Camera unavailable', 'NotFoundError');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } }, audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Video element unavailable');
      video.srcObject = stream; await video.play();
      const visionModuleUrl = VISION_URL;
      const visionModule = (await import(/* @vite-ignore */ visionModuleUrl)) as {
        FilesetResolver: { forVisionTasks: (root: string) => Promise<unknown> };
        FaceLandmarker: { createFromOptions: (fileset: unknown, options: object) => Promise<FaceLandmarkerLike> };
      };
      const fileset = await visionModule.FilesetResolver.forVisionTasks(WASM_URL);
      const commonOptions = {
        runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.55, minFacePresenceConfidence: 0.55, minTrackingConfidence: 0.5,
      };
      try {
        landmarkerRef.current = await visionModule.FaceLandmarker.createFromOptions(fileset, {
          ...commonOptions, baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        });
      } catch {
        landmarkerRef.current = await visionModule.FaceLandmarker.createFromOptions(fileset, {
          ...commonOptions, baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
        });
      }
      setStatus('ready'); runExperience('camera');
    } catch (cameraError) {
      stopExperience(); setError(friendlyCameraError(cameraError)); setStatus('error');
    }
  }, [runExperience, status, stopExperience]);

  const startDemo = useCallback(() => {
    stopExperience(); setError(''); setStatus('demo'); setMood('neutral'); runExperience('demo');
  }, [runExperience, stopExperience]);

  useEffect(() => stopExperience, [stopExperience]);

  const moodCopy = {
    neutral: { label: '等待表情', hint: '看向镜头，然后试着微笑' },
    smile: { label: '检测到微笑', hint: '下雨啦，再张嘴大笑试试' },
    laugh: { label: '检测到大笑', hint: '烟花已触发，移动头部撞击粒子' },
  }[mood];

  return (
    <main className="app-shell">
      <section className="experience" aria-label="AR 表情互动演示">
        <video ref={videoRef} className={`camera ${status === 'ready' ? 'is-visible' : ''}`} playsInline muted />
        <div className={`ambient ${status === 'ready' ? 'is-hidden' : ''}`} aria-hidden="true">
          <span className="orb orb-one" /><span className="orb orb-two" /><span className="orb orb-three" />
          <div className="face-guide"><i /></div>
        </div>
        <canvas ref={canvasRef} className="effects" aria-hidden="true" />
        <div className="vignette" aria-hidden="true" />
        <header className="topbar">
          <a className="brand" href="#" aria-label="Smile Storm 首页"><span className="brand-mark">S</span><span>SMILE STORM</span></a>
          <button className="metric-toggle" type="button" onClick={() => setShowMetrics((value) => !value)} aria-pressed={showMetrics}><span className="live-dot" /> 性能 {showMetrics ? '收起' : '详情'}</button>
        </header>

        {(status === 'ready' || status === 'demo') && (
          <div className={`mood-card mood-${mood}`} role="status" aria-live="polite">
            <span className="mood-icon">{mood === 'laugh' ? '✦' : mood === 'smile' ? '⌁' : '◌'}</span>
            <div><strong>{moodCopy.label}</strong><small>{moodCopy.hint}</small></div>
          </div>
        )}
        {showMetrics && (
          <aside className="metrics" aria-label="实时性能数据">
            <div><span>渲染</span><b>{metrics.fps} FPS</b></div><div><span>推理</span><b>{metrics.inference} 次/秒</b></div>
            <div><span>粒子</span><b>{metrics.particles} / {metrics.tier === '节能' ? 220 : 420}</b></div><div><span>档位</span><b>{metrics.tier}</b></div>
            <p>识别与渲染均在本机完成，视频不会上传。</p>
          </aside>
        )}

        {(status === 'idle' || status === 'loading' || status === 'error') && (
          <div className="welcome-panel">
            <p className="eyebrow">LIVE AR EXPERIENCE</p><h1>用一个笑容<br />改变天气</h1>
            <p className="lead">微笑唤醒一场雨，大笑点亮烟花。<br className="desktop-break" />移动你的头，亲自撞开每一颗粒子。</p>
            {error && <p className="error-message" role="alert">{error}</p>}
            <div className="actions">
              <button className="primary-button" type="button" onClick={startCamera} disabled={status === 'loading'}>
                {status === 'loading' ? <><span className="spinner" /> 正在加载模型…</> : <>开启摄像头 <span>→</span></>}
              </button>
              <button className="secondary-button" type="button" onClick={startDemo} disabled={status === 'loading'}>不使用摄像头，先看演示</button>
            </div>
            <p className="privacy-note"><span>✓</span> 端侧实时处理 · 不采集、不上传、不存储视频</p>
          </div>
        )}

        {(status === 'ready' || status === 'demo') && (
          <div className="bottom-guide">
            <div><span className={mood === 'smile' ? 'active' : ''}>01</span><p><b>微笑</b><small>触发雨滴</small></p></div><i />
            <div><span className={mood === 'laugh' ? 'active' : ''}>02</span><p><b>大笑</b><small>绽放烟花</small></p></div><i />
            <div><span className={mood === 'laugh' ? 'active' : ''}>03</span><p><b>移动头部</b><small>碰撞粒子</small></p></div>
            <button type="button" onClick={() => { stopExperience(); setStatus('idle'); setMood('neutral'); }}>退出</button>
          </div>
        )}
        <div className="corner-label left">CAMERA / {status === 'ready' ? 'LIVE' : status === 'demo' ? 'SIMULATION' : 'OFF'}</div>
        <div className="corner-label right">INFERENCE / ON-DEVICE</div>
      </section>
    </main>
  );
}
