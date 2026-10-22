import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
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
  title: "NBA History RAG",
  description: "Eval-driven retrieval over pinned NBA-history Wikipedia Articles",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <nav className="mx-auto flex w-full max-w-3xl gap-4 px-4 pt-6 text-sm font-medium">
          <Link href="/" className="hover:underline underline-offset-4">
            Search
          </Link>
          <Link href="/results" className="hover:underline underline-offset-4">
            Results
          </Link>
        </nav>
        {children}
        <footer className="px-4 py-6 text-center text-xs text-muted-foreground">
          Article text from Wikipedia, licensed under{" "}
          <a
            href="https://creativecommons.org/licenses/by-sa/4.0/"
            className="underline underline-offset-4"
          >
            CC BY-SA 4.0
          </a>
          .
        </footer>
      </body>
    </html>
  );
}
