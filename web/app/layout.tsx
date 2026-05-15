import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Adaptive Distributed SGD Dashboard",
  description: "Interactive visualization for Parameter Server, Ring AllReduce, and adaptive distributed SGD simulations."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
