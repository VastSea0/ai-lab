import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gorsel Sinir Agi Sandbox",
  description: "Saf TypeScript ile ileri besleme ve geriye yayilim gorsellestirme"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
