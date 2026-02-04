/**
 * Local Development Mode
 * 
 * When LOCAL_DEV_MODE=true, the application bypasses Manus OAuth and Forge services,
 * allowing you to run the full dashboard locally without external dependencies.
 */

import type { User } from "../../drizzle/schema";

export const LOCAL_DEV_MODE = process.env.LOCAL_DEV_MODE === "true";

/**
 * Default local user for development mode
 * This user is automatically "logged in" when LOCAL_DEV_MODE is enabled
 */
export const LOCAL_DEV_USER: User = {
  id: 1,
  openId: "local-dev-user",
  name: "Local Developer",
  email: "dev@localhost",
  loginMethod: "local",
  role: "admin",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
};

/**
 * Check if we're in local development mode
 */
export function isLocalDevMode(): boolean {
  return LOCAL_DEV_MODE;
}

/**
 * Get the local dev user if in local mode, otherwise null
 */
export function getLocalDevUser(): User | null {
  return LOCAL_DEV_MODE ? LOCAL_DEV_USER : null;
}
