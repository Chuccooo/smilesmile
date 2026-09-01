# 笔试提交内容

## 1. Live Demo

[https://smile-storm-ar-2026.cccljoan.chatgpt.site/](https://smile-storm-ar-2026.cccljoan.chatgpt.site/)

体验建议：首次进入请允许摄像头；微笑触发雨，张嘴大笑触发烟花，移动头部可撞开粒子。若设备无摄像头，可点击演示模式验证完整交互。

## 2. Source Code

[https://github.com/Chuccooo/smilesmile](https://github.com/Chuccooo/smilesmile)

代码仓库内 README 已包含架构、运行方式、性能策略、隐私说明、边缘情况与已知限制。

## 3. Vibecoding 复盘（200 字以内）

最大陷阱是 AI 最初倾向“每帧推理 + 无上限粒子 + 实时渐变模糊”，会阻塞主线程并重复放烟花。我改为 12/20Hz 推理、170/300 粒子预算和动态降载；发光纹理预烘焙后复用，并以平滑、迟滞、上升沿和冷却控制触发，头部碰撞采用椭圆近似。

核心纠偏 Prompt：

> 不要以视觉效果最大化为唯一目标。请先给出每帧 CPU/GPU 成本预算；将人脸推理与渲染解耦，限制推理频率和粒子总量；表情触发必须使用平滑、迟滞阈值、上升沿与冷却时间；碰撞采用满足体验的最低成本近似，并覆盖低端手机、权限拒绝、丢脸和后台标签页。
