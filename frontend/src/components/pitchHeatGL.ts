// ─── WebGL GPGPU Heat Renderer ────────────────────────────────────────────────
// Cursor heat simulation using ping-pong framebuffers + fragment shaders.
// Same architecture as Tamber.music: GPU diffusion/decay, Gaussian capsule
// injection, per-pixel colour mapping. Zero JavaScript grid loops.

export interface HeatRenderer {
  /** Inject heat along a capsule from (prevX,prevY) to (x,y) — all in UV [0,1],
   *  Y must be flipped from DOM space: yUV = 1 - mouseY/h  */
  inject(
    xUV: number,
    yUV: number,
    prevXUV: number,
    prevYUV: number,
    radius: number,
    intensity: number
  ): void
  step(dt: number): void
  render(): void
  resize(w: number, h: number): void
  destroy(): void
}

// ─── GLSL Sources ─────────────────────────────────────────────────────────────

const VERT_SRC = `
attribute vec2 a_pos;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`

/** Combined inject + simulate shader: reads texA, writes texB.
 *  All pixels are processed in parallel on the GPU. */
const UPDATE_FRAG = `
precision highp float;
uniform sampler2D u_prev;
uniform vec2      u_simSize;
uniform vec2      u_cursor;
uniform vec2      u_prevCursor;
uniform float     u_radius;
uniform float     u_intensity;
uniform float     u_decay;
uniform float     u_diffuse;

float capsuleDist(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
  return length(pa - ba * h);
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_simSize;
  vec2 ts = 1.0 / u_simSize;

  // 5-point stencil diffusion
  float center = texture2D(u_prev, uv).r;
  float up     = texture2D(u_prev, uv + vec2(0.0,  ts.y)).r;
  float down   = texture2D(u_prev, uv - vec2(0.0,  ts.y)).r;
  float left   = texture2D(u_prev, uv - vec2(ts.x, 0.0)).r;
  float right  = texture2D(u_prev, uv + vec2(ts.x, 0.0)).r;

  float avg      = (up + down + left + right) * 0.25;
  float diffused = center + (avg - center) * u_diffuse;

  // Capsule Gaussian injection
  float d      = capsuleDist(uv, u_prevCursor, u_cursor);
  float inject = u_intensity * exp(-(d * d) / (u_radius * u_radius));

  float raw    = (diffused + inject) * u_decay;
  float result = raw < 0.015 ? 0.0 : clamp(raw, 0.0, 1.0);
  gl_FragColor = vec4(result, 0.0, 0.0, 1.0);
}
`

/** Render shader: 3-layer composite — subtle "reveal through darkness" aesthetic.
 *  1. Faint ambient heat tint (back)
 *  2. Grid lines softly revealed by heat (middle)
 *  3. Pitch line SDF gently illuminated by heat (front)
 *
 *  Design: Tamber.music-inspired — cursor unveils hidden structure,
 *  NOT a bright spotlight. No white-hot, all values kept low.
 */
const RENDER_FRAG = `
precision highp float;
uniform sampler2D u_heat;
uniform vec2      u_resolution;
uniform vec2      u_pitchOffset;
uniform float     u_pitchScale;
uniform float     u_gridSize;
uniform vec2      u_cursor;
uniform vec2      u_prevCursor;

// ── Pitch line SDF ──────────────────────────────────────────────
// Coordinates in pitch-space (0-105 x 0-68), matching SVG viewBox.

float segDistH(vec2 p, float y, float x0, float x1) {
  float cx = clamp(p.x, x0, x1);
  return length(p - vec2(cx, y));
}

float segDistV(vec2 p, float x, float y0, float y1) {
  float cy = clamp(p.y, y0, y1);
  return length(p - vec2(x, cy));
}

float pitchSDF(vec2 p) {
  float d = 1e6;

  // Outer boundary (0,0 → 105,68)
  d = min(d, segDistH(p, 0.0,   0.0,  105.0));
  d = min(d, segDistH(p, 68.0,  0.0,  105.0));
  d = min(d, segDistV(p, 0.0,   0.0,  68.0));
  d = min(d, segDistV(p, 105.0, 0.0,  68.0));

  // Center line + circle
  d = min(d, segDistV(p, 52.5, 0.0, 68.0));
  d = min(d, abs(length(p - vec2(52.5, 34.0)) - 9.15));

  // Left penalty box (0,24.84 → 16.5,43.16)
  d = min(d, segDistH(p, 24.84, 0.0,  16.5));
  d = min(d, segDistH(p, 43.16, 0.0,  16.5));
  d = min(d, segDistV(p, 16.5,  24.84, 43.16));

  // Right penalty box (88.5,24.84 → 105,43.16)
  d = min(d, segDistH(p, 24.84, 88.5, 105.0));
  d = min(d, segDistH(p, 43.16, 88.5, 105.0));
  d = min(d, segDistV(p, 88.5,  24.84, 43.16));

  // Left goal box (0,13.84 → 5.5,54.16)
  d = min(d, segDistH(p, 13.84, 0.0, 5.5));
  d = min(d, segDistH(p, 54.16, 0.0, 5.5));
  d = min(d, segDistV(p, 5.5,   13.84, 54.16));

  // Right goal box (99.5,13.84 → 105,54.16)
  d = min(d, segDistH(p, 13.84, 99.5, 105.0));
  d = min(d, segDistH(p, 54.16, 99.5, 105.0));
  d = min(d, segDistV(p, 99.5,  13.84, 54.16));

  return d;
}

// ── Grid line distance (screen-space pixels) ───────────────────
float gridDist(vec2 sp) {
  float gx = min(mod(sp.x, u_gridSize), u_gridSize - mod(sp.x, u_gridSize));
  float gy = min(mod(sp.y, u_gridSize), u_gridSize - mod(sp.y, u_gridSize));
  return min(gx, gy);
}

// ── Alpha-over compositing ──────────────────────────────────────
vec4 over(vec4 front, vec4 back) {
  return front + back * (1.0 - front.a);
}

// ── Capsule distance for football shape mask ────────────────────
float capsuleDistFoot(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
  return length(pa - ba * h);
}

void main() {
  vec2  uv = gl_FragCoord.xy / u_resolution;
  // Screen pixel coords with Y=0 at top (matches CSS/SVG space)
  vec2  sp = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
  float heat = texture2D(u_heat, uv).r;

  // ── Layer 0: Pixel cells (uniform size, fill hot area) ─────────
  float CELL    = 22.0;
  vec2  cellIdx = floor(sp / CELL);
  vec2  cellCtr = (cellIdx + 0.5) * CELL;
  vec2  chUV    = clamp(vec2(cellCtr.x / u_resolution.x, 1.0 - cellCtr.y / u_resolution.y), 0.0, 1.0);
  float ch      = texture2D(u_heat, chUV).r;
  vec2  loc     = sp - cellIdx * CELL;
  float cellHalf = CELL * 0.5;
  float cellExp  = mix(0.46, 0.88, smoothstep(0.03, 0.80, ch));
  float halfSz   = cellHalf * cellExp;
  float inSq     = step(abs(loc.x - cellHalf), halfSz) * step(abs(loc.y - cellHalf), halfSz);
  vec3  coldCell = vec3(0.039, 0.271, 0.122);
  vec3  hotCell  = vec3(0.133, 0.773, 0.369);
  vec3  cellCol  = mix(coldCell, hotCell, smoothstep(0.03, 0.65, ch));
  float outerFade  = smoothstep(0.04, 0.14, ch);
  float innerFade  = 1.0 - smoothstep(0.35, 0.55, ch);
  float cellAlpha  = outerFade * innerFade * 0.58 * inSq;
  vec4  cellColor = vec4(cellCol, cellAlpha);

  // ── Layer 1: Faint ambient heat tint ──────────────────────────
  vec4 heatColor = vec4(0.0);
  if (heat > 0.02) {
    float t    = pow(heat, 0.7);
    vec3  tint = vec3(0.010, 0.220, 0.060);
    vec3  mid  = vec3(0.086, 0.773, 0.220);
    vec3  col  = mix(tint, mid, smoothstep(0.0, 0.7, t));
    float tintRing = 1.0 - smoothstep(0.30, 0.55, heat);
    float a    = smoothstep(0.02, 0.18, t) * 0.14 * tintRing;
    heatColor  = vec4(col, a);
  }

  // ── Layer 2: Grid lines ────────────────────────────────────────
  // Cold: 8% opacity (more visible at rest). Hot: 30% (revealed).
  float gd        = gridDist(sp);
  float gridLine  = 1.0 - smoothstep(0.0, 1.2, gd);
  float gridHeat  = smoothstep(0.03, 0.35, heat);
  float gridAlpha = gridLine * mix(0.08, 0.30, gridHeat);
  vec3  gridCol   = mix(
    vec3(0.133, 0.773, 0.369),   // green-500 cold
    vec3(0.337, 0.922, 0.557),   // green-300 warm
    gridHeat
  );
  vec4 gridColor = vec4(gridCol, gridAlpha);

  // ── Layer 3: Pitch line SDF ────────────────────────────────────
  // Cold: 22% opacity (clearly visible). Hot: 65%.
  vec2  pitchP    = (sp - u_pitchOffset) / u_pitchScale;
  float pd        = pitchSDF(pitchP) * u_pitchScale;
  float pitchLine = 1.0 - smoothstep(0.0, 1.5, pd);
  float pitchHeat = smoothstep(0.0, 0.30, heat);
  vec3  pitchCol  = mix(
    vec3(0.780, 1.000, 0.840),
    vec3(0.337, 0.922, 0.557),
    smoothstep(0.0, 0.6, pitchHeat)
  );
  float pitchAlpha = pitchLine * mix(0.22, 0.65, pitchHeat);
  vec4  pitchColor = vec4(pitchCol, pitchAlpha);

  // ── Composite back→front ────────────────────────────────────────
  vec4 result = over(cellColor, heatColor);
  result = over(gridColor, result);
  result = over(pitchColor, result);

  if (result.a < 0.002) discard;
  gl_FragColor = result;
}
`

// ─── GL Helpers ───────────────────────────────────────────────────────────────

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s)
    gl.deleteShader(s)
    throw new Error(`Shader error: ${log}`)
  }
  return s
}

function makeProgram(gl: WebGLRenderingContext, vert: string, frag: string): WebGLProgram {
  const p = gl.createProgram()!
  gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vert))
  gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, frag))
  gl.linkProgram(p)
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(p)}`)
  }
  return p
}

function makeTex(gl: WebGLRenderingContext, w: number, h: number): WebGLTexture {
  const t = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, t)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  return t
}

function makeFBO(gl: WebGLRenderingContext, tex: WebGLTexture): WebGLFramebuffer {
  const fbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
  const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`FBO incomplete: ${st}`)
  return fbo
}

// ─── Pitch mapping helper ─────────────────────────────────────────────────────
// Replicates SVG preserveAspectRatio="xMidYMid meet" for the 105x68 pitch.

const PITCH_W = 105,
  PITCH_H = 68

function computePitchMapping(canvasW: number, canvasH: number) {
  const scale = Math.min(canvasW / PITCH_W, canvasH / PITCH_H)
  return {
    scale,
    offX: (canvasW - PITCH_W * scale) / 2,
    offY: (canvasH - PITCH_H * scale) / 2,
  }
}

// ─── WebGL renderer ───────────────────────────────────────────────────────────

export function createHeatRenderer(canvas: HTMLCanvasElement): HeatRenderer {
  const gl = (canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: false,
    antialias: false,
  }) ??
    canvas.getContext('experimental-webgl', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
    })) as WebGLRenderingContext | null

  if (!gl) return createFallbackRenderer(canvas)

  let updateProg: WebGLProgram
  let renderProg: WebGLProgram
  try {
    updateProg = makeProgram(gl, VERT_SRC, UPDATE_FRAG)
    renderProg = makeProgram(gl, VERT_SRC, RENDER_FRAG)
  } catch {
    return createFallbackRenderer(canvas)
  }

  // Cache uniform locations — never call getUniformLocation in the hot path
  const uLoc = {
    prev: gl.getUniformLocation(updateProg, 'u_prev'),
    simSize: gl.getUniformLocation(updateProg, 'u_simSize'),
    cursor: gl.getUniformLocation(updateProg, 'u_cursor'),
    prevCursor: gl.getUniformLocation(updateProg, 'u_prevCursor'),
    radius: gl.getUniformLocation(updateProg, 'u_radius'),
    intensity: gl.getUniformLocation(updateProg, 'u_intensity'),
    decay: gl.getUniformLocation(updateProg, 'u_decay'),
    diffuse: gl.getUniformLocation(updateProg, 'u_diffuse'),
  }
  const rLoc = {
    heat: gl.getUniformLocation(renderProg, 'u_heat'),
    resolution: gl.getUniformLocation(renderProg, 'u_resolution'),
    pitchOffset: gl.getUniformLocation(renderProg, 'u_pitchOffset'),
    pitchScale: gl.getUniformLocation(renderProg, 'u_pitchScale'),
    gridSize: gl.getUniformLocation(renderProg, 'u_gridSize'),
    cursor: gl.getUniformLocation(renderProg, 'u_cursor'),
    prevCursor: gl.getUniformLocation(renderProg, 'u_prevCursor'),
  }
  const uPosLoc = gl.getAttribLocation(updateProg, 'a_pos')
  const rPosLoc = gl.getAttribLocation(renderProg, 'a_pos')

  // Fullscreen quad (triangle strip)
  const quad = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, quad)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)

  let canvasW = canvas.width
  let canvasH = canvas.height
  // Simulation runs at 1/4 resolution — linear upsampling gives natural blur
  let simW = Math.max(1, Math.ceil(canvasW / 4))
  let simH = Math.max(1, Math.ceil(canvasH / 4))

  let texA = makeTex(gl, simW, simH)
  let texB = makeTex(gl, simW, simH)
  let fboA = makeFBO(gl, texA)
  let fboB = makeFBO(gl, texB)

  let readTex = texA
  let writeTex = texB
  let writeFBO = fboB

  // Pitch mapping (updated in resize)
  let pitchMapping = computePitchMapping(canvasW, canvasH)

  // Pending inject state (cleared each step)
  let curX = 0.5,
    curY = 0.5,
    prevX = 0.5,
    prevY = 0.5
  let injectR = 0.065,
    injectI = 0.0

  const drawQuad = (posLoc: number) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, quad)
    gl.enableVertexAttribArray(posLoc)
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }

  return {
    inject(xUV, yUV, pxUV, pyUV, radius, intensity) {
      curX = xUV
      curY = yUV
      prevX = pxUV
      prevY = pyUV
      injectR = radius
      injectI = intensity
    },

    step(dt) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO)
      gl.viewport(0, 0, simW, simH)
      gl.useProgram(updateProg)

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, readTex)
      gl.uniform1i(uLoc.prev, 0)
      gl.uniform2f(uLoc.simSize, simW, simH)
      gl.uniform2f(uLoc.cursor, curX, curY)
      gl.uniform2f(uLoc.prevCursor, prevX, prevY)
      gl.uniform1f(uLoc.radius, injectR)
      gl.uniform1f(uLoc.intensity, injectI)
      gl.uniform1f(uLoc.decay, Math.pow(0.91, dt * 60))
      gl.uniform1f(uLoc.diffuse, 0.14)

      drawQuad(uPosLoc)

      // Reset injection — next frame has no heat unless inject() called
      injectI = 0.0

      // Swap ping-pong
      ;[readTex, writeTex] = [writeTex, readTex]
      writeFBO = writeTex === texA ? fboA : fboB
    },

    render() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvasW, canvasH)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.useProgram(renderProg)

      // LINEAR filtering for smooth 4x upsampling
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, readTex)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

      gl.uniform1i(rLoc.heat, 0)
      gl.uniform2f(rLoc.resolution, canvasW, canvasH)
      gl.uniform2f(rLoc.pitchOffset, pitchMapping.offX, pitchMapping.offY)
      gl.uniform1f(rLoc.pitchScale, pitchMapping.scale)
      gl.uniform1f(rLoc.gridSize, 80.0)
      gl.uniform2f(rLoc.cursor, curX, curY)
      gl.uniform2f(rLoc.prevCursor, prevX, prevY)

      drawQuad(rPosLoc)

      // Restore NEAREST for the simulation step
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    },

    resize(w, h) {
      canvasW = w
      canvasH = h
      simW = Math.max(1, Math.ceil(w / 4))
      simH = Math.max(1, Math.ceil(h / 4))
      pitchMapping = computePitchMapping(w, h)

      gl.deleteTexture(texA)
      gl.deleteTexture(texB)
      gl.deleteFramebuffer(fboA)
      gl.deleteFramebuffer(fboB)

      texA = makeTex(gl, simW, simH)
      texB = makeTex(gl, simW, simH)
      fboA = makeFBO(gl, texA)
      fboB = makeFBO(gl, texB)
      readTex = texA
      writeTex = texB
      writeFBO = fboB
    },

    destroy() {
      gl.deleteTexture(texA)
      gl.deleteTexture(texB)
      gl.deleteFramebuffer(fboA)
      gl.deleteFramebuffer(fboB)
      gl.deleteBuffer(quad)
      gl.deleteProgram(updateProg)
      gl.deleteProgram(renderProg)
    },
  }
}

// ─── Canvas 2D fallback (no WebGL) ────────────────────────────────────────────

export function createFallbackRenderer(canvas: HTMLCanvasElement): HeatRenderer {
  const ctx = canvas.getContext('2d')
  if (!ctx) return createNullRenderer()

  const CELL = 20
  let w = canvas.width,
    h = canvas.height
  let cols = Math.ceil(w / CELL) + 2
  let rows = Math.ceil(h / CELL) + 2
  let data = new Float32Array(cols * rows)

  let cxPx = -1,
    cyPx = -1,
    pxPx = -1,
    pyPx = -1
  let iRadius = 2.0,
    iIntensity = 0.0

  const off = document.createElement('canvas')
  off.width = w
  off.height = h
  const offCtx = off.getContext('2d')!

  return {
    inject(xUV, yUV, pxUV, pyUV, radius, intensity) {
      cxPx = xUV * w
      cyPx = (1 - yUV) * h
      pxPx = pxUV * w
      pyPx = (1 - pyUV) * h
      iRadius = (radius * Math.min(w, h)) / CELL
      iIntensity = intensity
    },

    step(dt) {
      if (iIntensity > 0) {
        const dx = cxPx - pxPx,
          dy = cyPx - pyPx
        const dist = Math.sqrt(dx * dx + dy * dy)
        const steps = Math.max(1, Math.ceil(dist / (CELL * 0.5)))
        for (let s = 0; s <= steps; s++) {
          const t = s / steps
          const px = pxPx + dx * t,
            py = pyPx + dy * t
          const cc = Math.floor(px / CELL),
            cr = Math.floor(py / CELL)
          const r = Math.ceil(iRadius)
          for (let row = Math.max(0, cr - r); row <= Math.min(rows - 1, cr + r); row++) {
            for (let col = Math.max(0, cc - r); col <= Math.min(cols - 1, cc + r); col++) {
              const dc = col - px / CELL,
                dr = row - py / CELL
              const d = Math.sqrt(dc * dc + dr * dr)
              const falloff = Math.exp(-(d * d) / (iRadius * iRadius))
              data[row * cols + col] = Math.min(
                1,
                data[row * cols + col] + (falloff * iIntensity) / (steps + 1)
              )
            }
          }
        }
        iIntensity = 0
      }

      const decay = Math.pow(0.91, dt * 60)
      const tmp = new Float32Array(data.length)
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const i = row * cols + col
          const c = data[i]
          const u = row > 0 ? data[(row - 1) * cols + col] : c
          const d = row < rows - 1 ? data[(row + 1) * cols + col] : c
          const l = col > 0 ? data[row * cols + col - 1] : c
          const ri = col < cols - 1 ? data[row * cols + col + 1] : c
          tmp[i] = c + ((u + d + l + ri) * 0.25 - c) * 0.06
        }
      }
      for (let i = 0; i < data.length; i++) {
        data[i] = Math.min(1, Math.max(0, tmp[i] * decay))
      }
    },

    render() {
      ctx.clearRect(0, 0, w, h)
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const raw = data[row * cols + col]
          // Cold structural dots (always visible for ambient texture)
          if (raw < 0.05) {
            if ((col + row) % 2 === 0) {
              ctx.fillStyle = 'rgba(34,197,94,0.08)'
              ctx.fillRect(col * CELL + CELL / 2 - 1, row * CELL + CELL / 2 - 1, 2, 2)
            }
            continue
          }
          // Border band only: skip the hot core (> 0.60), only render the perimeter ring
          if (raw > 0.6) continue
          const outerFade = Math.min(1, (raw - 0.05) / 0.13)
          const innerFade = 1 - Math.min(1, Math.max(0, (raw - 0.38) / 0.22))
          const borderMask = outerFade * innerFade
          if (borderMask < 0.01) continue
          // Taper: cells at tail are tiny, cells near cursor are large
          const taper = Math.min(1, Math.max(0, (raw - 0.05) / 0.47))
          const sz = CELL * (0.18 + taper * 0.66)
          const offset = (CELL - sz) / 2
          const t = Math.min(1, (raw - 0.04) / 0.24)
          // Deep teal-emerald → vivid emerald (#10B981)
          const g = Math.round(97 + t * (209 - 97))
          const r2 = Math.round(5 + t * (16 - 5))
          const b2 = Math.round(66 + t * (130 - 66))
          const a = (borderMask * 0.82).toFixed(3)
          ctx.fillStyle = `rgba(${r2},${g},${b2},${a})`
          ctx.fillRect(col * CELL + offset, row * CELL + offset, sz, sz)
        }
      }
    },

    resize(nw, nh) {
      w = nw
      h = nh
      cols = Math.ceil(w / CELL) + 2
      rows = Math.ceil(h / CELL) + 2
      data = new Float32Array(cols * rows)
      off.width = w
      off.height = h
    },

    destroy() {
      /* nothing to release */
    },
  }
}

function createNullRenderer(): HeatRenderer {
  return { inject() {}, step() {}, render() {}, resize() {}, destroy() {} }
}
