import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '笑遇星雨 · 表情互动 AR',
  description: '微笑落雨，大笑绽放烟花，情绪触发雨与烟火。所有识别均在浏览器端完成。',
  openGraph: {
    title: '笑遇星雨 · 表情互动 AR',
    description: '微笑落雨，大笑绽放烟花。端侧实时人脸表情识别与粒子物理互动原型。',
    type: 'website',
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
