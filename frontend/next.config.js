/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // ./run.sh serves the dashboard with `next dev`; keep its badge off the demo.
  devIndicators: false,
};

module.exports = nextConfig;
