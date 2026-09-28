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

/* Applied before the first paint, which is the whole point: a stored choice
   read in useEffect lands after the browser has already painted the other
   theme, and the flash is exactly what that looks like. No stored choice
   leaves data-theme unset, so the media query in globals.css follows the OS.
   Wrapped in try/catch because a browser with site data blocked throws on
   the localStorage read, and a theme script must never break the page. */
const NO_FLASH = `try{var t=localStorage.getItem("pitwall-theme");if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t)}}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${text.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH }} />
      </head>
      <body>
        <Nav />
        <main className="mx-auto max-w-[1320px] px-5 pb-20 sm:px-8">{children}</main>
      </body>
    </html>
  );
}
