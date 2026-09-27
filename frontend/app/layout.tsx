import type { Metadata } from "next";
import { IBM_Plex_Sans, Saira_Condensed } from "next/font/google";
import Nav from "@/components/Nav";
import "./globals.css";

/* Two families, self-hosted at build by next/font — no runtime CDN request.
   Saira Condensed carries headlines and figures (timing-screen vernacular);
   IBM Plex Sans carries everything else and has real tabular figures. */
const display = Saira_Condensed({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

const text = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-text",
  display: "swap",
});

export const metadata: Metadata = {
  title: "F1 Race Strategy Intelligence",
  description: "Race-engineering analytics for the 2023 Bahrain Grand Prix — computational intelligence over one real session.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${text.variable}`}>
      <body>
        <Nav />
        <main className="mx-auto max-w-[1320px] px-5 pb-20 sm:px-8">{children}</main>
      </body>
    </html>
  );
}
