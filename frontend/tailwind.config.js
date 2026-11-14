/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#070a0e',
          900: '#0d1117',
          850: '#131922',
          800: '#161b22',
          700: '#21262d',
          600: '#30363d',
        },
        gold: {
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
        },
        bull: {
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
        },
        bear: {
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
        },
      },
    },
  },
  plugins: [],
};
