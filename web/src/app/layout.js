import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Providers from "@/components/Providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "LeafInsight - Disease Detection",
  description: "Comparative Explainable Deep Learning Platform for Indoor Plant Leaf Disease Detection",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "LeafInsight",
  },
  openGraph: {
    images: ['/og.png'],
  },
};

export const viewport = {
  themeColor: "#0f172a",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen bg-slate-50 text-slate-900 antialiased`}>
        <Providers>
          <div className="relative flex min-h-screen flex-col pb-24 md:pb-0">
            <Navbar />
            <main className="flex-1">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
