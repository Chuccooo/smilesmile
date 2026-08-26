import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Smile Storm · 表情互动 AR',
  description: '微笑触发雨滴，大笑绽放烟花，并用头部与粒子实时碰撞。所有识别均在浏览器端完成。',
  openGraph: {
    title: 'Smile Storm · 表情互动 AR',
    description: '用一个笑容改变天气。端侧实时人脸表情识别与粒子物理互动原型。',
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
