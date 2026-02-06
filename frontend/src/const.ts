export { COOKIE_NAME, ONE_YEAR_MS } from "@/shared/const";

// Check if running in local dev mode (no OAuth configured)
export const isLocalDevMode = !import.meta.env.VITE_OAUTH_PORTAL_URL || import.meta.env.VITE_OAUTH_PORTAL_URL === "";

// Generate login URL at runtime so redirect URI reflects the current origin.
export const getLoginUrl = () => {
  // In local dev mode, there's no OAuth - user is auto-logged in
  if (isLocalDevMode) {
    return "/";
  }

  const oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL;
  const appId = import.meta.env.VITE_APP_ID;
  const redirectUri = `${window.location.origin}/api/oauth/callback`;
  const state = btoa(redirectUri);

  const url = new URL(`${oauthPortalUrl}/app-auth`);
  url.searchParams.set("appId", appId);
  url.searchParams.set("redirectUri", redirectUri);
  url.searchParams.set("state", state);
  url.searchParams.set("type", "signIn");

  return url.toString();
};
