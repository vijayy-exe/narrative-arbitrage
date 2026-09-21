import type { Metadata } from "next";
import { Geist, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Narrative Arbitrage",
  description: "Live paper-trading dashboard for the narrative-arbitrage project (Phase 7)",
};

const NAV_LINKS = [
  { href: "/", label: "Assets" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/research", label: "Research" },
  { href: "/lab", label: "Lab" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${jetbrainsMono.variable} h-full antialiased`}
      style={{ colorScheme: "dark" }}
    >
      <body className="min-h-full flex flex-col" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        <header
          className="sticky top-0 z-10 backdrop-blur-sm"
          style={{ borderBottom: "1px solid var(--border)", background: "color-mix(in srgb, var(--background) 85%, transparent)" }}
        >
          <nav className="mx-auto max-w-5xl px-6 py-3 flex items-center gap-6 text-sm">
            <Link href="/" className="font-semibold tracking-tight flex items-center gap-2">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: "var(--color-gain)", boxShadow: "0 0 6px var(--color-gain)" }}
                aria-hidden
              />
              Narrative Arbitrage
            </Link>
            <div className="flex gap-5">
              {NAV_LINKS.slice(1).map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="transition-colors"
                  style={{ color: "var(--foreground-muted)" }}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </nav>
        </header>
        <div className="flex-1">{children}</div>
      </body>
    </html>
  );
}
