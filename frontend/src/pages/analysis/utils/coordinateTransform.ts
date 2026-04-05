export const PITCH_W = 105
export const PITCH_H = 68

const DEFAULT_VIDEO_W = 1920
const DEFAULT_VIDEO_H = 1080

/**
 * Map video pixel coordinates to pitch SVG units (0–105 x 0–68).
 * Player positions come from bbox center-bottom (foot contact point).
 */
export function pixelToPitch(
  px: number,
  py: number,
  videoWidth = DEFAULT_VIDEO_W,
  videoHeight = DEFAULT_VIDEO_H
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(PITCH_W, (px / videoWidth) * PITCH_W)),
    y: Math.max(0, Math.min(PITCH_H, (py / videoHeight) * PITCH_H)),
  }
}

/**
 * Convert ball pitchPos (centimetres from export_tracks_json) to pitch SVG units.
 * pitchPos values are in cm (0–10500 x 0–6800).
 */
export function ballCmToPitch(pitchCm: [number, number]): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(PITCH_W, pitchCm[0] / 100)),
    y: Math.max(0, Math.min(PITCH_H, pitchCm[1] / 100)),
  }
}
