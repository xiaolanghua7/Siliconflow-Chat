# SiliconFlow Chat Tool

一个基于 `tkinter` 的本地桌面聊天工具，用来调用硅基流动（SiliconFlow）模型广场里的模型进行对话。

## 功能

- 输入 API Key 和模型名称后直接聊天
- 支持添加、切换、删除多个模型
- 支持单独设置聊天模型和摘要模型
- 支持自定义系统提示词模板
- 支持生成参数调节
- 支持长对话自动摘要，保留短期上下文
- 支持导出聊天记录
- 支持本地保存窗口位置、模型列表、模板、参数和历史记录
- Windows 下启用 DPI 适配，界面更清晰

## 文件说明

- `test.py`：启动入口
- `siliconflow_chat_app.py`：主程序
- `siliconflow_chat_config.json`：本地配置文件和聊天历史

## 环境要求

- Python 3.10+
- `openai` Python 包
- Windows 自带的 `tkinter` 一般可直接使用

安装依赖：

```bash
pip install openai
```

## 运行方式

直接运行：

```bash
python test.py
```

也可以直接运行：

```bash
python siliconflow_chat_app.py
```

## 使用步骤

1. 打开 `Account` 页，填写 `API Key`
2. 如有需要，修改 `Base URL`，默认是 `https://api.siliconflow.cn/v1`
3. 在 `Models` 页输入模型名称，点 `Add Model`
4. 选择要作为聊天模型的条目
5. 如需独立压缩长记忆，可以再选一个摘要模型
6. 在 `Templates` 页选择或编辑系统提示词
7. 在 `Params` 页调整 `temperature`、`top_p`、`max_tokens` 等参数
8. 在 `Memory` 页开启自动摘要，设置触发阈值和保留最近消息数
9. 在右侧聊天框输入内容，点击 `Send`，或用 `Ctrl+Enter`

## 本地存储

程序会把配置写入同目录下的 `siliconflow_chat_config.json`，包括：

- API Key
- 模型列表
- 当前模型
- 提示词模板
- 生成参数
- 记忆设置
- 聊天历史
- 窗口位置和大小

这只会占用很少的磁盘空间。主要增长点是你本地保存的聊天历史，通常也只是普通 JSON 文本，不会占用很多 C 盘空间。

## 实用建议

- 想要更稳、更准确：把 `temperature` 调低
- 想要更长回答：把 `max_tokens` 调高
- 想要长对话更不容易丢信息：开启自动摘要，并把 `keep_recent_messages` 设为 6 到 12 左右
- 想要更适合复杂任务：给不同场景准备不同模板，例如 `Coding`、`Deep Reasoning`、`Writing`
- 如果摘要模型和聊天模型分开，通常可以更好地平衡速度和上下文管理

## 备注

- API Key 会保存在本地配置文件里，适合个人电脑本地使用
- 如果想彻底重置配置，删除 `siliconflow_chat_config.json` 即可

