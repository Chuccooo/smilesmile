'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

type Status = 'idle' | 'loading' | 'ready' | 'error' | 'demo';
type Mood = 'neutral' | 'smile' | 'laugh';

type Particle = {
  x: number; y: number; px: number; py: number; vx: number; vy: number;
  radius: number; life: number; maxLife: number;
  kind: 'rain' | 'spark' | 'ember' | 'splash' | 'willow' | 'star';
  tone: number; phase: number; trail: number; collided: boolean;
  delay?: number; depth?: 0 | 1 | 2; style?: 0 | 1 | 2;
  brightness?: number; rotation?: number; starVariant?: 0 | 1 | 2;
};
type Burst = {
  x: number; y: number; life: number; maxLife: number; tone: number; rotation: number;
  kind: 'firework' | 'water' | 'curtain'; delay?: number; style?: 0 | 1 | 2;
  brightness?: number; scale?: number; spokes?: number;
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
const COLORS = ['#edfaff', '#8de4ff', '#88a8ff', '#b89cff', '#77e4dc', '#ffd0b4'];

function rgba(hex: string, alpha: number) {
  const value = hex.replace('#', '');
  const red = Number.parseInt(value.slice(0, 2), 16);
  const green = Number.parseInt(value.slice(2, 4), 16);
  const blue = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${red},${green},${blue},${alpha})`;
}

function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function expressionThresholds(sensitivity: number) {
  const value = clamp(sensitivity);
  const smileEnter = 0.54 - value * 0.28;
  return {
    smileEnter,
    smileExit: Math.max(0.14, smileEnter - 0.11),
    laughEnter: 0.82 - value * 0.34,
    laughExit: 0.62 - value * 0.2,
    jawEnter: 0.38 - value * 0.22,
    jawExit: 0.25 - value * 0.1,
  };
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
  const sensitivityRef = useRef(0.72);
  const fireworkDensityRef = useRef(1);
  const [status, setStatus] = useState<Status>('idle');
  const [mood, setMood] = useState<Mood>('neutral');
  const [error, setError] = useState('');
  const [showMetrics, setShowMetrics] = useState(false);
  const [sensitivity, setSensitivity] = useState(72);
  const [fireworkDensity, setFireworkDensity] = useState(100);
  const [metrics, setMetrics] = useState({
    fps: 0, inference: 0, particles: 0, budget: 300, tier: '影院',
    smile: 0, jaw: 0, smileThreshold: 0.34, laughThreshold: 0.58,
  });

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
    const baseParticleBudget = lowPower ? 170 : 300;
    let particleBudget = baseParticleBudget;
    let qualityScale = 1;
    let slowSamples = 0;
    let fastSamples = 0;
    const inferenceInterval = lowPower ? 84 : 50;
    const collider: FaceCollider = { cx: 0, cy: 0, rx: 0, ry: 0, visible: false };
    const particles: Particle[] = [];
    const bursts: Burst[] = [];
    let width = 1;
    let height = 1;
    let lastFrame = performance.now();
    const experienceStartedAt = lastFrame;
    let lastInference = 0;
    let lastMetricUpdate = performance.now();
    let framesForMetric = 0;
    let inferenceCount = 0;
    let smileActive = false;
    let laughActive = false;
    let lastFirework = -3000;
    let laughStartedAt = 0;
    let waterfallTriggered = false;
    let waterfallUntil = 0;
    let rainCarry = 0;
    let smoothSmile = 0;
    let smoothLaugh = 0;
    let smoothJaw = 0;
    let currentMood: Mood = 'neutral';

    // These tiny glow textures are baked once. Every live particle reuses them,
    // avoiding hundreds of radial-gradient and shadow-blur calculations per frame.
    const glowSprites = COLORS.map((color) => {
      const sprite = document.createElement('canvas');
      sprite.width = 80; sprite.height = 80;
      const spriteContext = sprite.getContext('2d');
      if (!spriteContext) return sprite;
      const glow = spriteContext.createRadialGradient(40, 40, 0, 40, 40, 40);
      glow.addColorStop(0, '#ffffff');
      glow.addColorStop(0.08, rgba(color, 1));
      glow.addColorStop(0.28, rgba(color, 0.72));
      glow.addColorStop(0.62, rgba(color, 0.16));
      glow.addColorStop(1, rgba(color, 0));
      spriteContext.fillStyle = glow;
      spriteContext.fillRect(0, 0, 80, 80);
      return sprite;
    });

    // Three baked star silhouettes replace coarse white dots. Size, rotation and
    // alpha vary at runtime, while the expensive gradients stay cached.
    const starSprites = COLORS.map((color) => [0, 1, 2].map((variant) => {
      const sprite = document.createElement('canvas');
      sprite.width = 96; sprite.height = 96;
      const spriteContext = sprite.getContext('2d');
      if (!spriteContext) return sprite;
      const halo = spriteContext.createRadialGradient(48, 48, 0, 48, 48, 42);
      halo.addColorStop(0, 'rgba(255,255,255,1)');
      halo.addColorStop(0.07, rgba(color, 0.96));
      halo.addColorStop(0.24, rgba(color, 0.42));
      halo.addColorStop(1, rgba(color, 0));
      spriteContext.globalAlpha = 0.62;
      spriteContext.fillStyle = halo;
      spriteContext.fillRect(0, 0, 96, 96);
      spriteContext.translate(48, 48);
      spriteContext.globalCompositeOperation = 'lighter';
      const axes = variant === 0 ? 2 : variant === 1 ? 3 : 4;
      for (let axis = 0; axis < axes; axis += 1) {
        const angle = axis * Math.PI / axes;
        const length = axis === 0 ? 38 : variant === 0 ? 17 : 21 + (axis % 2) * 7;
        const ray = spriteContext.createLinearGradient(-length, 0, length, 0);
        ray.addColorStop(0, rgba(color, 0));
        ray.addColorStop(0.4, rgba(color, 0.58));
        ray.addColorStop(0.5, 'rgba(255,255,255,.98)');
        ray.addColorStop(0.6, rgba(color, 0.58));
        ray.addColorStop(1, rgba(color, 0));
        spriteContext.save();
        spriteContext.rotate(angle);
        spriteContext.strokeStyle = ray;
        spriteContext.lineWidth = axis === 0 ? 1.15 : 0.68;
        spriteContext.beginPath();
        spriteContext.moveTo(-length, 0);
        spriteContext.quadraticCurveTo(0, axis % 2 ? 0.5 : -0.35, length, 0);
        spriteContext.stroke();
        spriteContext.restore();
      }
      spriteContext.fillStyle = 'rgba(255,255,255,.98)';
      spriteContext.beginPath(); spriteContext.arc(0, 0, 1.45, 0, Math.PI * 2); spriteContext.fill();
      return sprite;
    }));

    const flareSprite = (() => {
      const sprite = document.createElement('canvas');
      sprite.width = 160; sprite.height = 160;
      const spriteContext = sprite.getContext('2d');
      if (!spriteContext) return sprite;
      const glow = spriteContext.createRadialGradient(80, 80, 0, 80, 80, 80);
      glow.addColorStop(0, 'rgba(255,255,255,1)');
      glow.addColorStop(0.05, 'rgba(217,248,255,.96)');
      glow.addColorStop(0.2, 'rgba(126,188,255,.46)');
      glow.addColorStop(0.55, 'rgba(144,112,255,.14)');
      glow.addColorStop(1, 'rgba(92,80,220,0)');
      spriteContext.fillStyle = glow;
      spriteContext.fillRect(0, 0, 160, 160);
      spriteContext.translate(80, 80);
      spriteContext.strokeStyle = 'rgba(255,255,255,.9)';
      spriteContext.lineCap = 'round';
      [[58, 1.2], [35, 2.2], [22, 3.2]].forEach(([length, widthValue], index) => {
        spriteContext.save();
        spriteContext.rotate((Math.PI / 4) * index);
        spriteContext.lineWidth = widthValue;
        spriteContext.beginPath();
        spriteContext.moveTo(-length, 0); spriteContext.lineTo(length, 0);
        spriteContext.moveTo(0, -length); spriteContext.lineTo(0, length);
        spriteContext.stroke();
        spriteContext.restore();
      });
      return sprite;
    })();

    const curtainSprite = (() => {
      const sprite = document.createElement('canvas');
      sprite.width = 96; sprite.height = 256;
      const spriteContext = sprite.getContext('2d');
      if (!spriteContext) return sprite;
      const glow = spriteContext.createLinearGradient(0, 0, 0, 256);
      glow.addColorStop(0, 'rgba(132,226,255,.44)');
      glow.addColorStop(0.16, 'rgba(86,151,255,.28)');
      glow.addColorStop(0.48, 'rgba(111,87,255,.12)');
      glow.addColorStop(1, 'rgba(60,52,190,0)');
      spriteContext.fillStyle = glow;
      spriteContext.fillRect(0, 0, 96, 256);
      return sprite;
    })();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, lowPower ? 1.5 : 2);
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
      bursts.push({
        x: originX, y: originY, life: 0.64, maxLife: 0.64,
        tone: 2, rotation: Math.random() * Math.PI, kind: 'firework',
      });

      const density = fireworkDensityRef.current;
      const outerCount = Math.round((lowPower ? 72 : 126) * qualityScale * density);
      const innerCount = Math.round((lowPower ? 38 : 64) * qualityScale * density);
      const dustCount = Math.round((lowPower ? 24 : 46) * qualityScale * density);
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));

      for (let index = 0; index < outerCount && particles.length < particleBudget; index += 1) {
        const angle = index * goldenAngle + Math.random() * 0.018;
        const wave = 0.82 + Math.sin(index * 0.41) * 0.12;
        const speed = (142 + Math.random() * 128) * wave;
        const tone = index % 9 === 0 ? 3 : index % 4 === 0 ? 2 : 1;
        particles.push({
          x: originX, y: originY, px: originX, py: originY,
          vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
          radius: 1.05 + Math.random() * 1.45, life: 1.45 + Math.random() * 1.1,
          maxLife: 2.55, kind: 'spark', tone, phase: Math.random() * Math.PI * 2,
          trail: 0.88 + Math.random() * 0.25, collided: false,
        });
      }

      for (let index = 0; index < innerCount && particles.length < particleBudget; index += 1) {
        const angle = index / innerCount * Math.PI * 2 + Math.sin(index * 1.7) * 0.055;
        const speed = 72 + Math.random() * 96;
        particles.push({
          x: originX, y: originY, px: originX, py: originY,
          vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
          radius: 0.8 + Math.random() * 1.1, life: 1.1 + Math.random() * 0.85,
          maxLife: 1.95, kind: 'spark', tone: index % 3 === 0 ? 3 : 2,
          phase: Math.random() * Math.PI * 2, trail: 0.62, collided: false,
        });
      }

      for (let index = 0; index < dustCount && particles.length < particleBudget; index += 1) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 25 + Math.random() * 150;
        particles.push({
          x: originX, y: originY, px: originX, py: originY,
          vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
          radius: 0.35 + Math.random() * 0.7, life: 0.6 + Math.random() * 1.25,
          maxLife: 1.85, kind: 'ember', tone: Math.random() > 0.7 ? 2 : 0,
          phase: Math.random() * Math.PI * 2, trail: 0.28, collided: false,
        });
      }
    };

    const spawnWaterfall = (now: number) => {
      waterfallUntil = now + 1980;

      // Reuse the existing device budget for the two-second hero effect. Clearing
      // previous rain/fireworks prevents an invisible budget spike at the handoff.
      particles.length = 0;
      bursts.length = 0;

      bursts.push({
        x: width * 0.5, y: 0, life: 1.98, maxLife: 1.98,
        tone: 1, rotation: 0, kind: 'curtain',
      });

      const emitterCount = lowPower ? 6 : 8;
      const desiredCount = Math.round((lowPower ? 138 : 244) * qualityScale * fireworkDensityRef.current);
      const availableCount = Math.max(0, Math.min(desiredCount, particleBudget - particles.length));
      const starCount = Math.min(Math.round(availableCount * 0.18), lowPower ? 26 : 44);
      const willowCount = Math.max(0, availableCount - starCount);
      const perEmitter = Math.max(1, Math.ceil(willowCount / emitterCount));
      const styleOffset = Math.floor(Math.random() * 3);
      let created = 0;

      const pickTone = () => {
        const roll = Math.random();
        if (roll < 0.34) return 1;
        if (roll < 0.55) return 2;
        if (roll < 0.69) return 3;
        if (roll < 0.81) return 0;
        if (roll < 0.93) return 4;
        return 5;
      };

      for (let emitter = 0; emitter < emitterCount && created < willowCount; emitter += 1) {
        const pairRank = Math.floor(Math.abs(emitter - (emitterCount - 1) / 2));
        const emitterDelay = pairRank * 0.1 + Math.random() * 0.018;
        const style = ((emitter + styleOffset) % 3) as 0 | 1 | 2;
        const scale = style === 0 ? 0.96 + Math.random() * 0.26 : style === 1 ? 0.8 + Math.random() * 0.32 : 0.74 + Math.random() * 0.22;
        const brightness = Math.min(1.12, 0.56 + Math.random() * 0.48 + (pairRank === 0 ? 0.08 : 0));
        const originX = width * ((emitter + 0.5) / emitterCount) + (Math.random() - 0.5) * width * 0.035;
        const originY = height * (0.09 + pairRank * 0.018 + Math.random() * 0.012);
        bursts.push({
          x: originX, y: originY, life: 0.56, maxLife: 0.56,
          tone: emitter % 5 === 0 ? 5 : emitter % 3 === 0 ? 3 : emitter % 2 === 0 ? 2 : 1,
          rotation: Math.random() * Math.PI, kind: 'firework', delay: emitterDelay,
          style, brightness, scale, spokes: style === 0 ? 14 : style === 1 ? 18 : 10,
        });

        for (let strand = 0; strand < perEmitter && created < willowCount && particles.length < particleBudget; strand += 1) {
          const normalized = perEmitter <= 1 ? 0.5 : strand / (perEmitter - 1);
          const depthRoll = Math.random();
          const depth = (depthRoll < 0.3 ? 0 : depthRoll < 0.8 ? 1 : 2) as 0 | 1 | 2;
          const styleSpread = style === 0 ? 0.8 : style === 1 ? 0.98 : 0.58;
          const spread = styleSpread * (depth === 0 ? 1.06 : depth === 1 ? 0.94 : 0.82);
          const angle = -Math.PI * (0.5 + spread / 2) + normalized * Math.PI * spread + (Math.random() - 0.5) * 0.09;
          const speedFactor = style === 0 ? 1 : style === 1 ? 1.12 : 0.86;
          const speed = (92 + Math.sin(normalized * Math.PI) * 92 + Math.random() * 42) * speedFactor * (0.84 + depth * 0.12);
          const lifeOffset = style === 0 ? 0.1 : style === 1 ? -0.1 : 0.15;
          const life = Math.min(1.92 - emitterDelay, 1.62 + lifeOffset - depth * 0.08 + Math.random() * 0.12);
          particles.push({
            x: originX, y: originY, px: originX, py: originY,
            vx: Math.cos(angle) * speed * 0.74,
            vy: Math.sin(angle) * speed * 0.92,
            radius: (0.62 + depth * 0.2 + Math.random() * 0.88) * (style === 1 ? 0.88 : 1),
            life, maxLife: life, kind: 'willow',
            tone: pickTone(),
            phase: Math.random() * Math.PI * 2,
            trail: 0.42 + Math.random() * 0.42,
            collided: false, delay: emitterDelay + depth * 0.025 + Math.random() * 0.09,
            depth, style, brightness: Math.min(1.15, brightness * (0.66 + Math.random() * 0.5)),
            rotation: Math.random() * Math.PI,
          });
          created += 1;
        }
      }

      const starTones = [0, 3, 0, 4, 2, 0, 5, 1];
      for (let index = 0; index < starCount && particles.length < particleBudget; index += 1) {
        const depthRoll = Math.random();
        const depth = (depthRoll < 0.58 ? 0 : depthRoll < 0.9 ? 1 : 2) as 0 | 1 | 2;
        const life = 0.78 + Math.random() * 0.86;
        particles.push({
          x: Math.random() * width, y: height * (0.06 + Math.random() * 0.66),
          px: 0, py: 0, vx: (Math.random() - 0.5) * (3 + depth * 4), vy: 3 + depth * 5 + Math.random() * 7,
          radius: depth === 0 ? 0.48 + Math.random() * 0.4 : depth === 1 ? 0.78 + Math.random() * 0.58 : 1.25 + Math.random() * 0.65,
          life, maxLife: life, kind: 'star', tone: starTones[index % starTones.length],
          phase: Math.random() * Math.PI * 2, trail: 0, collided: false,
          delay: 0.05 + Math.random() * 0.92, depth,
          brightness: depth === 0 ? 0.18 + Math.random() * 0.28 : depth === 1 ? 0.46 + Math.random() * 0.36 : 0.86 + Math.random() * 0.28,
          rotation: Math.random() * Math.PI, starVariant: (index % 3) as 0 | 1 | 2,
        });
      }
    };

    const spawnRain = (dt: number) => {
      rainCarry += dt * (lowPower ? 38 : 68) * qualityScale;
      const amount = Math.min(5, Math.floor(rainCarry));
      rainCarry -= amount;
      for (let index = 0; index < amount && particles.length < particleBudget; index += 1) {
        const x = Math.random() * width; const y = -20 - Math.random() * 70;
        const depth = 0.58 + Math.random() * 0.72;
        particles.push({
          x, y, px: x, py: y, vx: (-10 + Math.random() * 6) * depth, vy: (310 + Math.random() * 230) * depth,
          radius: 0.35 + depth * 0.52, life: 2.9, maxLife: 2.9, kind: 'rain', tone: Math.random() > 0.72 ? 2 : 1,
          phase: Math.random() * Math.PI * 2, trail: 0.62 + depth * 0.38, collided: false,
        });
      }
    };

    const spawnSplash = (x: number, y: number, energy: number) => {
      const count = Math.round((lowPower ? 5 : 8) * qualityScale);
      for (let index = 0; index < count && particles.length < particleBudget; index += 1) {
        const angle = -Math.PI + Math.random() * Math.PI;
        const speed = (48 + Math.random() * 125) * (0.75 + energy * 0.35);
        particles.push({
          x, y, px: x, py: y, vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 25, radius: 0.55 + Math.random() * 0.8,
          life: 0.38 + Math.random() * 0.38, maxLife: 0.76, kind: 'splash',
          tone: Math.random() > 0.8 ? 2 : 1, phase: Math.random() * Math.PI * 2,
          trail: 0.42, collided: false,
        });
      }
      if (bursts.length < 36) bursts.push({ x, y, life: 0.36, maxLife: 0.36, tone: 1, rotation: 0, kind: 'water' });
    };

    const collideWithHead = (particle: Particle) => {
      if (!collider.visible) return false;
      const expandedRx = collider.rx + particle.radius;
      const expandedRy = collider.ry + particle.radius;
      const dx = particle.x - collider.cx; const dy = particle.y - collider.cy;
      const value = (dx * dx) / (expandedRx * expandedRx) + (dy * dy) / (expandedRy * expandedRy);
      if (value >= 1) return false;
      particle.collided = true;
      const length = Math.sqrt(value) || 0.001;
      const nx = dx / (expandedRx * expandedRx * length);
      const ny = dy / (expandedRy * expandedRy * length);
      const normalLength = Math.hypot(nx, ny) || 1;
      const ux = nx / normalLength; const uy = ny / normalLength;
      const boundaryScale = 1 / length;
      particle.x = collider.cx + dx * boundaryScale;
      particle.y = collider.cy + dy * boundaryScale;
      if (particle.kind === 'rain') return true;
      const velocityAlongNormal = particle.vx * ux + particle.vy * uy;
      if (velocityAlongNormal < 0) {
        const bounce = particle.kind === 'willow' ? 0.48 : particle.kind === 'ember' ? 0.56 : 0.74;
        particle.vx -= (1 + bounce) * velocityAlongNormal * ux;
        particle.vy -= (1 + bounce) * velocityAlongNormal * uy;
      }
      particle.vx += ux * 44;
      return true;
    };

    const updateMood = (smile: number, jawOpen: number, now: number) => {
      smoothSmile += (smile - smoothSmile) * 0.32;
      smoothJaw += (jawOpen - smoothJaw) * 0.34;
      const rawLaugh = clamp(smile * 0.72 + jawOpen * 0.62);
      smoothLaugh += (rawLaugh - smoothLaugh) * 0.34;
      const thresholds = expressionThresholds(sensitivityRef.current);
      if (!smileActive && smoothSmile >= thresholds.smileEnter) smileActive = true;
      if (smileActive && smoothSmile <= thresholds.smileExit) smileActive = false;
      if (!laughActive && smoothLaugh >= thresholds.laughEnter && smoothJaw >= thresholds.jawEnter) {
        laughActive = true;
        laughStartedAt = now;
        waterfallTriggered = false;
        spawnFirework(now);
      }
      if (laughActive && !waterfallTriggered && now - laughStartedAt >= 5000) {
        waterfallTriggered = true;
        spawnWaterfall(now);
      }
      if (laughActive && (smoothLaugh <= thresholds.laughExit || smoothJaw <= thresholds.jawExit)) {
        laughActive = false;
        laughStartedAt = 0;
        waterfallTriggered = false;
      }
      const nextMood: Mood = laughActive ? 'laugh' : smileActive ? 'smile' : 'neutral';
      if (nextMood !== currentMood) { currentMood = nextMood; setMood(nextMood); }
    };

    const infer = (now: number) => {
      if (mode === 'demo') {
        const cycle = ((now - experienceStartedAt) / 1000) % 16;
        const demoSmile = cycle > 1.2 && cycle < 13.8 ? 0.68 : 0.08;
        const demoJaw = cycle > 3.2 && cycle < 10.2 ? 0.78 : 0.05;
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
      if (smileActive && now >= waterfallUntil) spawnRain(dt);
      context.clearRect(0, 0, width, height);

      // Physics update stays allocation-free; rendering happens in grouped passes below.
      for (let index = particles.length - 1; index >= 0; index -= 1) {
        const particle = particles[index];
        if ((particle.delay ?? 0) > 0) {
          particle.delay = Math.max(0, (particle.delay ?? 0) - dt);
          particle.px = particle.x; particle.py = particle.y;
          continue;
        }
        particle.px = particle.x; particle.py = particle.y; particle.life -= dt;
        particle.phase += dt * (particle.kind === 'ember' ? 8 : particle.kind === 'splash' ? 7 : particle.kind === 'willow' ? 5.5 : particle.kind === 'star' ? 6.5 : 4.5);
        const styleLift = particle.kind === 'willow' ? particle.style === 0 ? 18 : particle.style === 1 ? -16 : 30 : 0;
        particle.vy += (particle.kind === 'rain' ? 520 : particle.kind === 'ember' ? 155 : particle.kind === 'splash' ? 280 : particle.kind === 'willow' ? 292 + styleLift : particle.kind === 'star' ? 12 : 105) * dt;
        particle.vx *= Math.pow(particle.kind === 'rain' ? 0.994 : particle.kind === 'ember' ? 0.972 : particle.kind === 'splash' ? 0.982 : particle.kind === 'willow' ? 0.987 : particle.kind === 'star' ? 0.994 : 0.986, dt * 60);
        particle.x += particle.vx * dt; particle.y += particle.vy * dt;
        if (particle.kind === 'rain' && collideWithHead(particle)) {
          spawnSplash(particle.x, particle.y, Math.min(1.4, particle.vy / 450));
          particles.splice(index, 1);
          continue;
        }
        if (particle.kind === 'spark' || particle.kind === 'ember' || particle.kind === 'willow') collideWithHead(particle);
        if (particle.life <= 0 || particle.y > height + 80 || particle.x < -100 || particle.x > width + 100) {
          particles.splice(index, 1); continue;
        }
      }

      context.save();
      context.globalCompositeOperation = 'lighter';
      context.lineCap = 'round';

      // Layer 1: baked bloom, rotating flare and expanding shockwaves.
      for (let index = bursts.length - 1; index >= 0; index -= 1) {
        const burst = bursts[index];
        if ((burst.delay ?? 0) > 0) {
          burst.delay = Math.max(0, (burst.delay ?? 0) - dt);
          continue;
        }
        burst.life -= dt;
        if (burst.life <= 0) { bursts.splice(index, 1); continue; }
        const progress = 1 - burst.life / burst.maxLife;
        const eased = 1 - Math.pow(1 - progress, 3);
        const fade = Math.pow(1 - progress, 1.7);
        if (burst.kind === 'curtain') {
          const envelope = Math.min(1, progress * 9) * Math.pow(1 - progress, 0.72);
          context.globalAlpha = envelope * 0.72;
          context.drawImage(curtainSprite, 0, 0, width, height * 0.82);
          continue;
        }
        if (burst.kind === 'water') {
          context.globalAlpha = fade * 0.72;
          context.strokeStyle = 'rgba(161,231,255,.92)';
          context.lineWidth = 0.75;
          context.beginPath();
          context.ellipse(burst.x, burst.y, 2 + eased * 28, 0.8 + eased * 7, 0, 0, Math.PI * 2);
          context.stroke();
          continue;
        }
        const brightness = burst.brightness ?? 1;
        const burstScale = burst.scale ?? 1;
        const bloomSize = (120 + eased * 290) * burstScale;
        context.globalAlpha = fade * 0.82 * brightness;
        context.drawImage(flareSprite, burst.x - bloomSize / 2, burst.y - bloomSize / 2, bloomSize, bloomSize);
        context.save();
        context.translate(burst.x, burst.y); context.rotate(burst.rotation + progress * 0.7);
        const flareSize = (86 + eased * 120) * burstScale;
        context.globalAlpha = fade * 0.72 * brightness;
        context.drawImage(flareSprite, -flareSize / 2, -flareSize / 2, flareSize, flareSize);
        context.restore();
        context.globalAlpha = fade * 0.72 * brightness;
        context.strokeStyle = rgba(COLORS[burst.tone], 0.92);
        if (burst.style === undefined) {
          context.lineWidth = 1.2 + fade * 2.4;
          context.beginPath(); context.arc(burst.x, burst.y, 14 + eased * 145, 0, Math.PI * 2); context.stroke();
          context.globalAlpha = fade * 0.32 * brightness;
          context.lineWidth = 1;
          context.beginPath(); context.arc(burst.x, burst.y, 8 + eased * 210, 0, Math.PI * 2); context.stroke();
        } else {
          const arcStart = burst.style === 0 ? -Math.PI * 0.9 : burst.style === 1 ? -Math.PI * 1.01 : -Math.PI * 0.76;
          const arcEnd = burst.style === 0 ? -Math.PI * 0.1 : burst.style === 1 ? Math.PI * 0.01 : -Math.PI * 0.24;
          const inner = (8 + eased * 34) * burstScale;
          const outer = (14 + eased * 128) * burstScale;
          context.lineWidth = 0.65 + fade * 1.5;
          context.beginPath(); context.arc(burst.x, burst.y, outer, arcStart, arcEnd); context.stroke();
          context.globalAlpha = fade * 0.42 * brightness;
          context.lineWidth = 0.56;
          context.beginPath();
          const spokes = burst.spokes ?? 12;
          for (let spoke = 0; spoke < spokes; spoke += 1) {
            const ratio = spoke / Math.max(1, spokes - 1);
            const angle = arcStart + ratio * (arcEnd - arcStart) + Math.sin(burst.rotation * 7 + spoke * 4.13) * 0.025;
            const variation = 0.72 + (0.5 + 0.5 * Math.sin(burst.rotation * 11 + spoke * 9.17)) * 0.34;
            context.moveTo(burst.x + Math.cos(angle) * inner, burst.y + Math.sin(angle) * inner);
            context.quadraticCurveTo(
              burst.x + Math.cos(angle + 0.025) * outer * 0.64 * variation,
              burst.y + Math.sin(angle + 0.025) * outer * 0.64 * variation,
              burst.x + Math.cos(angle + 0.055) * outer * variation,
              burst.y + Math.sin(angle + 0.055) * outer * variation,
            );
          }
          context.stroke();
          if (burst.style === 2) {
            context.globalAlpha = fade * 0.2 * brightness;
            context.beginPath(); context.arc(burst.x, burst.y, outer * 1.18, arcStart + 0.08, arcEnd - 0.08); context.stroke();
          }
        }
      }

      // Layer 2: rain is batched into two luminous passes instead of one shadow blur per drop.
      context.globalAlpha = 0.18;
      context.strokeStyle = 'rgba(47,169,255,.8)';
      context.lineWidth = 2.4;
      context.beginPath();
      for (const particle of particles) {
        if (particle.kind !== 'rain') continue;
        const length = Math.min(18, Math.abs(particle.vy) * 0.026 * particle.trail);
        context.moveTo(particle.x - particle.vx * 0.025, particle.y - length);
        context.lineTo(particle.x, particle.y + 2);
      }
      context.stroke();
      context.globalAlpha = 0.78;
      context.strokeStyle = 'rgba(209,248,255,.94)';
      context.lineWidth = 0.58;
      context.beginPath();
      for (const particle of particles) {
        if (particle.kind !== 'rain') continue;
        const length = Math.min(14, Math.abs(particle.vy) * 0.022 * particle.trail);
        context.moveTo(particle.x - particle.vx * 0.02, particle.y - length);
        context.lineTo(particle.x, particle.y + 1.5);
      }
      context.stroke();

      context.globalAlpha = 0.62;
      context.strokeStyle = 'rgba(178,239,255,.92)';
      context.lineWidth = 0.7;
      context.beginPath();
      for (const particle of particles) {
        if (particle.kind !== 'splash') continue;
        context.moveTo(particle.px, particle.py);
        context.lineTo(particle.x, particle.y);
      }
      context.stroke();

      // Layer 3: short curved willow trails taper in three batched segments.
      // Grouping by depth and tone keeps the richer look below the device budget.
      const segmentStarts = [1, 0.66, 0.32];
      const segmentEnds = [0.66, 0.32, 0];
      const segmentAlpha = [0.14, 0.38, 0.82];
      const segmentWidth = [0.26, 0.52, 0.92];
      for (let depth = 0; depth < 3; depth += 1) {
        const depthAlpha = depth === 0 ? 0.3 : depth === 1 ? 0.68 : 1;
        const depthWidth = depth === 0 ? 0.58 : depth === 1 ? 0.88 : 1.2;
        for (let tone = 0; tone < COLORS.length; tone += 1) {
          context.strokeStyle = rgba(COLORS[tone], 0.94);

          context.globalAlpha = depthAlpha * 0.12;
          context.lineWidth = depthWidth * 4.6;
          context.beginPath();
          for (const particle of particles) {
            if (particle.kind !== 'willow' || particle.tone !== tone || (particle.depth ?? 1) !== depth || (particle.delay ?? 0) > 0) continue;
            const tailTime = 0.055 + particle.trail * 0.075;
            const wave = Math.sin(particle.phase) * (1.2 + depth * 0.8);
            const tailX = particle.x - particle.vx * tailTime + wave;
            const tailY = particle.y - particle.vy * tailTime;
            const midX = particle.x - particle.vx * tailTime * 0.48 - wave * 0.4;
            const midY = particle.y - particle.vy * tailTime * 0.48;
            context.moveTo(tailX, tailY);
            context.quadraticCurveTo(midX, midY, particle.x, particle.y);
          }
          context.stroke();

          for (let segment = 0; segment < 3; segment += 1) {
            context.globalAlpha = depthAlpha * segmentAlpha[segment];
            context.lineWidth = depthWidth * segmentWidth[segment];
            context.beginPath();
            for (const particle of particles) {
              if (particle.kind !== 'willow' || particle.tone !== tone || (particle.depth ?? 1) !== depth || (particle.delay ?? 0) > 0) continue;
              const tailTime = 0.055 + particle.trail * 0.075;
              const start = segmentStarts[segment];
              const end = segmentEnds[segment];
              const wave = Math.sin(particle.phase + segment * 1.4) * (0.7 + depth * 0.7);
              const startX = particle.x - particle.vx * tailTime * start + wave * start;
              const startY = particle.y - particle.vy * tailTime * start;
              const endX = particle.x - particle.vx * tailTime * end - wave * end * 0.35;
              const endY = particle.y - particle.vy * tailTime * end;
              context.moveTo(startX, startY);
              context.quadraticCurveTo((startX + endX) / 2 - wave * 0.35, (startY + endY) / 2, endX, endY);
              if ((particle.style === 0 || particle.phase > 5.2) && segment < 2) {
                const branch = (segment + 1) * (depth + 1) * 0.52;
                context.moveTo(startX + branch, startY);
                context.quadraticCurveTo((startX + endX) / 2 + branch * 1.4, (startY + endY) / 2, endX, endY);
              }
            }
            context.stroke();
          }
        }
      }

      // Layer 4: grouped colored comet trails need only two strokes per palette tone.
      for (let tone = 0; tone < COLORS.length; tone += 1) {
        context.globalAlpha = 0.22;
        context.strokeStyle = rgba(COLORS[tone], 0.82);
        context.lineWidth = 5.2;
        context.beginPath();
        for (const particle of particles) {
          if (particle.kind === 'rain' || particle.kind === 'splash' || particle.kind === 'willow' || particle.kind === 'star' || particle.tone !== tone || (particle.delay ?? 0) > 0) continue;
          const trailTime = (particle.kind === 'spark' ? 0.075 : 0.035) * particle.trail;
          context.moveTo(particle.x, particle.y);
          context.lineTo(particle.x - particle.vx * trailTime, particle.y - particle.vy * trailTime);
        }
        context.stroke();
        context.globalAlpha = 0.78;
        context.lineWidth = 1.05;
        context.beginPath();
        for (const particle of particles) {
          if (particle.kind === 'rain' || particle.kind === 'splash' || particle.kind === 'willow' || particle.kind === 'star' || particle.tone !== tone || (particle.delay ?? 0) > 0) continue;
          const trailTime = (particle.kind === 'spark' ? 0.065 : 0.03) * particle.trail;
          context.moveTo(particle.x, particle.y);
          context.lineTo(particle.x - particle.vx * trailTime, particle.y - particle.vy * trailTime);
        }
        context.stroke();
      }

      // Layer 5: draw cached glow sprites. Twinkle is alpha-only, so no new gradients.
      for (const particle of particles) {
        if (particle.kind === 'rain' || (particle.delay ?? 0) > 0) continue;
        const fade = clamp(particle.life / Math.min(particle.maxLife, particle.kind === 'ember' ? 0.42 : 0.7));
        const twinkle = particle.kind === 'spark' ? 0.72 + Math.sin(particle.phase) * 0.28 : particle.kind === 'willow' ? 0.62 + Math.sin(particle.phase) * 0.28 : particle.kind === 'star' ? 0.5 + Math.sin(particle.phase) * 0.5 : 0.48 + Math.sin(particle.phase) * 0.2;
        const collisionBoost = particle.collided ? 1.65 : 1;
        const depthScale = particle.depth === 0 ? 0.68 : particle.depth === 2 ? 1.2 : 0.94;
        const size = particle.radius * (particle.kind === 'spark' ? 9 : particle.kind === 'willow' ? 7 : particle.kind === 'star' ? 18 : 6) * collisionBoost * depthScale;
        context.globalAlpha = fade * twinkle * (particle.brightness ?? 1);
        if (particle.kind === 'star') {
          context.save();
          context.translate(particle.x, particle.y);
          context.rotate((particle.rotation ?? 0) + Math.sin(particle.phase * 0.4) * 0.06);
          context.drawImage(starSprites[particle.tone][particle.starVariant ?? 0], -size / 2, -size / 2, size, size);
          context.restore();
        } else if (particle.kind === 'willow' && (particle.tone === 0 || particle.phase > 5.55)) {
          context.save();
          context.translate(particle.x, particle.y);
          context.rotate((particle.rotation ?? 0) + particle.phase * 0.035);
          context.drawImage(starSprites[particle.tone][2], -size * 0.62, -size * 0.62, size * 1.24, size * 1.24);
          context.restore();
        } else {
          context.drawImage(glowSprites[particle.tone], particle.x - size / 2, particle.y - size / 2, size, size);
        }
        if (particle.collided && (particle.kind === 'spark' || particle.kind === 'willow')) {
          const flareSize = size * 2.6;
          context.globalAlpha = fade * 0.62;
          context.drawImage(flareSprite, particle.x - flareSize / 2, particle.y - flareSize / 2, flareSize, flareSize);
        }
        particle.collided = false;
      }
      context.restore();
      if (mode === 'demo' && collider.visible) {
        context.save(); context.setLineDash([7, 8]); context.lineWidth = 1.5;
        context.strokeStyle = 'rgba(255,255,255,.4)'; context.beginPath();
        context.ellipse(collider.cx, collider.cy, collider.rx, collider.ry, 0, 0, Math.PI * 2); context.stroke(); context.restore();
      }
      if (now - lastMetricUpdate > 1000) {
        const elapsed = Math.max(0.1, (now - lastMetricUpdate) / 1000);
        const measuredFps = Math.round(framesForMetric / elapsed);
        if (measuredFps < 43) { slowSamples += 1; fastSamples = 0; }
        else if (measuredFps > 56) { fastSamples += 1; slowSamples = 0; }
        else { slowSamples = 0; fastSamples = 0; }
        if (slowSamples >= 2) {
          qualityScale = Math.max(0.62, qualityScale - 0.12); slowSamples = 0;
          particleBudget = Math.round(baseParticleBudget * qualityScale);
          while (particles.length > particleBudget) {
            const expendable = particles.findLastIndex((particle) => particle.kind === 'star' || particle.kind === 'rain' || particle.kind === 'ember');
            particles.splice(expendable >= 0 ? expendable : particles.length - 1, 1);
          }
        } else if (fastSamples >= 3) {
          qualityScale = Math.min(1, qualityScale + 0.08); fastSamples = 0;
          particleBudget = Math.round(baseParticleBudget * qualityScale);
        }
        const thresholdSnapshot = expressionThresholds(sensitivityRef.current);
        setMetrics({
          fps: measuredFps, inference: Math.round(inferenceCount / elapsed), particles: particles.length,
          budget: particleBudget, tier: `${lowPower ? '节能精致' : '影院'} ${Math.round(qualityScale * 100)}%`,
          smile: smoothSmile, jaw: smoothJaw,
          smileThreshold: thresholdSnapshot.smileEnter, laughThreshold: thresholdSnapshot.laughEnter,
        });
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
    laugh: { label: '检测到大笑', hint: '烟花已触发，保持大笑 5 秒召唤烟花瀑布' },
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
          <button className="metric-toggle" type="button" onClick={() => setShowMetrics((value) => !value)} aria-pressed={showMetrics}><span className="live-dot" /> 灵敏度 · 性能 {showMetrics ? '收起' : '调参'}</button>
        </header>

        {(status === 'ready' || status === 'demo') && (
          <div className={`mood-card mood-${mood}`} role="status" aria-live="polite">
            <span className="mood-icon">{mood === 'laugh' ? '✦' : mood === 'smile' ? '⌁' : '◌'}</span>
            <div><strong>{moodCopy.label}</strong><small>{moodCopy.hint}</small></div>
          </div>
        )}
        {showMetrics && (
          <aside className="metrics" aria-label="表情灵敏度与实时性能数据">
            <label className="sensitivity-control">
              <span className="sensitivity-head"><span>表情灵敏度</span><b>{sensitivity}%</b></span>
              <input
                type="range" min="30" max="95" step="1" value={sensitivity}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setSensitivity(next); sensitivityRef.current = next / 100;
                }}
                aria-label="表情识别灵敏度"
              />
              <small>调高更容易触发；过高可能把说话误判为大笑</small>
            </label>
            <label className="sensitivity-control density-control">
              <span className="sensitivity-head"><span>烟花数量</span><b>{fireworkDensity}%</b></span>
              <input
                type="range" min="40" max="100" step="5" value={fireworkDensity}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setFireworkDensity(next); fireworkDensityRef.current = next / 100;
                }}
                aria-label="烟花粒子数量"
              />
              <small>控制普通烟花与瀑布密度；不会突破设备粒子预算</small>
            </label>
            <div><span>微笑分数 / 阈值</span><b>{metrics.smile.toFixed(2)} / {metrics.smileThreshold.toFixed(2)}</b></div>
            <div><span>张嘴分数</span><b>{metrics.jaw.toFixed(2)}</b></div>
            <div><span>大笑阈值</span><b>{metrics.laughThreshold.toFixed(2)}</b></div>
            <div><span>渲染</span><b>{metrics.fps} FPS</b></div><div><span>推理</span><b>{metrics.inference} 次/秒</b></div>
            <div><span>粒子</span><b>{metrics.particles} / {metrics.budget}</b></div><div><span>档位</span><b>{metrics.tier}</b></div>
            <p>星芒与发光纹理已预烘焙复用；数量滑块受设备预算和自动降载双重保护，视频不会上传。</p>
          </aside>
        )}

        {(status === 'idle' || status === 'loading' || status === 'error') && (
          <div className="welcome-panel">
            <p className="eyebrow">LIVE AR EXPERIENCE</p><h1>笑遇星雨</h1>
            <p className="lead">微笑落雨，大笑绽放烟花，情绪触发雨与烟火</p>
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
            <div><span className={mood === 'laugh' ? 'active' : ''}>03</span><p><b>持续大笑</b><small>5秒烟花瀑布</small></p></div>
            <button type="button" onClick={() => { stopExperience(); setStatus('idle'); setMood('neutral'); }}>退出</button>
          </div>
        )}
        <div className="corner-label left">CAMERA / {status === 'ready' ? 'LIVE' : status === 'demo' ? 'SIMULATION' : 'OFF'}</div>
        <div className="corner-label right">INFERENCE / ON-DEVICE</div>
      </section>
    </main>
  );
}
