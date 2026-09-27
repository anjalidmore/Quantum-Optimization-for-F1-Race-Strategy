/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Every colour resolves to a token in app/globals.css, so there is one
      // source of truth and Tailwind cannot drift from DESIGN.md.
      colors: {
        track: {
          000: "var(--track-000)",
          50: "var(--track-050)",
          100: "var(--track-100)",
          200: "var(--track-200)",
          300: "var(--track-300)",
        },
        paper: {
          900: "var(--paper-900)",
          700: "var(--paper-700)",
          500: "var(--paper-500)",
          400: "var(--paper-400)",
        },
        accent: "var(--accent)",
        "accent-ink": "var(--accent-ink)",
        edge: "var(--edge)",
        compound: {
          soft: "var(--compound-soft)",
          medium: "var(--compound-medium)",
          hard: "var(--compound-hard)",
          inter: "var(--compound-inter)",
          wet: "var(--compound-wet)",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "ui-sans-serif", "sans-serif"],
        text: ["var(--font-text)", "ui-sans-serif", "sans-serif"],
      },
      borderRadius: { sm: "var(--r-sm)", md: "var(--r-md)", lg: "var(--r-lg)" },
      maxWidth: { measure: "70ch" },
    },
  },
  plugins: [],
};
