/**
 * Storage Module
 * 
 * Supports two modes:
 * 1. Manus Forge Storage (production) - Uses cloud storage via Manus API
 * 2. Local File Storage (development) - Saves files to local filesystem
 * 
 * Set LOCAL_DEV_MODE=true to use local storage
 */

import { ENV } from './_core/env';
import { isLocalDevMode } from './_core/localMode';
import * as fs from 'fs';
import * as path from 'path';

// Local storage directory (relative to project root)
const LOCAL_STORAGE_DIR = process.env.LOCAL_STORAGE_DIR || './uploads';

type StorageConfig = { baseUrl: string; apiKey: string };

function getStorageConfig(): StorageConfig {
  const baseUrl = ENV.forgeApiUrl;
  const apiKey = ENV.forgeApiKey;

  if (!baseUrl || !apiKey) {
    throw new Error(
      "Storage proxy credentials missing: set BUILT_IN_FORGE_API_URL and BUILT_IN_FORGE_API_KEY"
    );
  }

  return { baseUrl: baseUrl.replace(/\/+$/, ""), apiKey };
}

function buildUploadUrl(baseUrl: string, relKey: string): URL {
  const url = new URL("v1/storage/upload", ensureTrailingSlash(baseUrl));
  url.searchParams.set("path", normalizeKey(relKey));
  return url;
}

async function buildDownloadUrl(
  baseUrl: string,
  relKey: string,
  apiKey: string
): Promise<string> {
  const downloadApiUrl = new URL(
    "v1/storage/downloadUrl",
    ensureTrailingSlash(baseUrl)
  );
  downloadApiUrl.searchParams.set("path", normalizeKey(relKey));
  const response = await fetch(downloadApiUrl, {
    method: "GET",
    headers: buildAuthHeaders(apiKey),
  });
  return (await response.json()).url;
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function normalizeKey(relKey: string): string {
  return relKey.replace(/^\/+/, "");
}

function toFormData(
  data: Buffer | Uint8Array | string,
  contentType: string,
  fileName: string
): FormData {
  const blob =
    typeof data === "string"
      ? new Blob([data], { type: contentType })
      : new Blob([data as any], { type: contentType });
  const form = new FormData();
  form.append("file", blob, fileName || "file");
  return form;
}

function buildAuthHeaders(apiKey: string): HeadersInit {
  return { Authorization: `Bearer ${apiKey}` };
}

// ============================================================================
// LOCAL STORAGE FUNCTIONS
// ============================================================================

function ensureLocalStorageDir(subPath: string): string {
  const fullPath = path.join(LOCAL_STORAGE_DIR, path.dirname(subPath));
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
  return path.join(LOCAL_STORAGE_DIR, subPath);
}

async function localStoragePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  _contentType = "application/octet-stream"
): Promise<{ key: string; url: string }> {
  const key = normalizeKey(relKey);
  const filePath = ensureLocalStorageDir(key);
  
  // Convert data to Buffer if needed
  const buffer = typeof data === "string" 
    ? Buffer.from(data) 
    : Buffer.from(data);
  
  fs.writeFileSync(filePath, buffer);
  
  // Return a local URL that can be served by Express
  const url = `/uploads/${key}`;
  
  console.log(`[Local Storage] Saved: ${filePath}`);
  return { key, url };
}

async function localStorageGet(relKey: string): Promise<{ key: string; url: string }> {
  const key = normalizeKey(relKey);
  const url = `/uploads/${key}`;
  return { key, url };
}

// ============================================================================
// CLOUD STORAGE FUNCTIONS (Manus Forge)
// ============================================================================

async function cloudStoragePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  contentType = "application/octet-stream"
): Promise<{ key: string; url: string }> {
  const { baseUrl, apiKey } = getStorageConfig();
  const key = normalizeKey(relKey);
  const uploadUrl = buildUploadUrl(baseUrl, key);
  const formData = toFormData(data, contentType, key.split("/").pop() ?? key);
  const response = await fetch(uploadUrl, {
    method: "POST",
    headers: buildAuthHeaders(apiKey),
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    throw new Error(
      `Storage upload failed (${response.status} ${response.statusText}): ${message}`
    );
  }
  const url = (await response.json()).url;
  return { key, url };
}

async function cloudStorageGet(relKey: string): Promise<{ key: string; url: string }> {
  const { baseUrl, apiKey } = getStorageConfig();
  const key = normalizeKey(relKey);
  return {
    key,
    url: await buildDownloadUrl(baseUrl, key, apiKey),
  };
}

// ============================================================================
// EXPORTED FUNCTIONS (Auto-switch based on mode)
// ============================================================================

export async function storagePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  contentType = "application/octet-stream"
): Promise<{ key: string; url: string }> {
  if (isLocalDevMode()) {
    return localStoragePut(relKey, data, contentType);
  }
  return cloudStoragePut(relKey, data, contentType);
}

export async function storageGet(relKey: string): Promise<{ key: string; url: string }> {
  if (isLocalDevMode()) {
    return localStorageGet(relKey);
  }
  return cloudStorageGet(relKey);
}

/**
 * Get the local storage directory path (for serving static files)
 */
export function getLocalStorageDir(): string {
  return LOCAL_STORAGE_DIR;
}
