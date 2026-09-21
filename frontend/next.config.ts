import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pins the workspace root to this frontend/ directory -- otherwise
  // Turbopack walks up looking for a lockfile and warns about picking up
  // the user's home-directory package-lock.json (harmless, but noisy).
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
