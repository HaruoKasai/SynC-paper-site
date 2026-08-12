import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://sync-paper-analysis.hkasaimd.chatgpt.site"),
  title: {
    default: "SynC | Analysis resources",
    template: "%s",
  },
  description:
    "Statistical methods, frozen source tables, and analysis code supporting the SynC manuscript.",
  openGraph: {
    title: "SynC analysis resources",
    description:
      "Statistical methods, frozen source tables, and analysis code supporting the SynC manuscript.",
    type: "website",
    siteName: "SynC analysis resources",
  },
  twitter: {
    card: "summary",
    title: "SynC analysis resources",
    description:
      "Statistical methods, frozen source tables, and analysis code supporting the SynC manuscript.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
