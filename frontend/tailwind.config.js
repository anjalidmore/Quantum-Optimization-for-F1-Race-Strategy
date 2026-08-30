/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        f1red: "#E10600",
        carbon: "#0d0f12",
        panel: "#15181d",
        panel2: "#1c2027",
      },
    },
  },
  plugins: [],
};
