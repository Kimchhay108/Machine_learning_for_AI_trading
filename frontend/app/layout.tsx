import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'XAUUSD Hybrid AI Trading Platform',
  description: 'Modular Quantitative AI Trading Research Platform focused on Gold (XAUUSD)',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-dark-950 text-gray-100 flex flex-col font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
