import withSerwistInit from "@serwist/next";

const withSerwist = withSerwistInit({
  swSrc: "src/app/sw.js",
  swDest: "public/sw.js",
  reloadOnOnline: true,
  disable: process.env.NODE_ENV === "development",
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {},
  /* config options here */
};

export default withSerwist(nextConfig);
