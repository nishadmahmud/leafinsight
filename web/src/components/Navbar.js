"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, BarChart3 } from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: '/', label: 'Dashboard', icon: Home },
    { href: '/metrics', label: 'Metrics', icon: BarChart3 },
  ];

  return (
    <>
      {/* Top Navbar (Desktop & Mobile Header) */}
      <nav className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/90 backdrop-blur-sm">
        <div className="container mx-auto flex h-12 items-center justify-between px-4">
          <Link href="/" className="flex items-center">
            <img src="/logo.png" alt="LeafInsight Logo" className="h-8 w-auto" />
          </Link>
          
          {/* Desktop Links */}
          <div className="hidden md:flex items-center gap-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-colors ${
                  pathname === link.href
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* Mobile Bottom Pill Nav */}
      <div className="md:hidden fixed bottom-5 left-1/2 -translate-x-1/2 z-50 w-[85%] max-w-[320px]">
        <nav className="flex items-center justify-around bg-slate-900/95 backdrop-blur-md rounded-full px-2 py-1.5 shadow-2xl shadow-slate-900/20 border border-slate-800">
          {links.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex flex-col items-center justify-center w-full py-1.5 rounded-full transition-all ${
                  isActive 
                    ? 'text-white bg-white/10' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
              >
                <Icon className={`h-[18px] w-[18px] ${isActive ? 'mb-0.5' : 'mb-1'}`} strokeWidth={isActive ? 2.5 : 2} />
                <span className={`text-[10px] font-medium ${isActive ? 'opacity-100' : 'opacity-70'}`}>
                  {link.label}
                </span>
              </Link>
            );
          })}
        </nav>
      </div>
    </>
  );
}
