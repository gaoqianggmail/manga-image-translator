/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  // Enable compression for better performance
  compress: true,
  // Optimize for production
  swcMinify: true,
}

module.exports = nextConfig