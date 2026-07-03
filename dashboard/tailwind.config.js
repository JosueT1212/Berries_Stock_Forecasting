/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#0b0e14", light: "#f7f8fa" },
        accent: { DEFAULT: "#6366f1", warm: "#f59e0b" },
      },
    },
  },
  darkMode: "media",
  plugins: [],
};
