# 多语言支持说明

本项目已实现完整的多语言支持系统，支持中英文切换。

## 功能特性

- ✅ 支持中英文双语
- ✅ 动态语言切换
- ✅ 节点显示名称多语言
- ✅ 错误消息多语言
- ✅ 状态消息多语言
- ✅ 调试消息多语言
- ✅ API提示词多语言
- ✅ 格式化字符串支持

## 文件结构

```
comfyui-ayang_node/
├── __init__.py                 # 主文件，包含多语言系统
├── locale/                     # 多语言文件目录
│   ├── en/                     # 英文翻译
│   │   └── strings.json        # 英文翻译文件
│   └── zh_CN/                  # 中文翻译
│       └── strings.json        # 中文翻译文件
├── locale_example.py           # 使用示例
└── LOCALE_README.md           # 本说明文件
```

## 使用方法

### 1. 基本使用

```python
from __init__ import set_locale, get_current_locale, locale_manager

# 设置语言为中文
set_locale("zh_CN")

# 设置语言为英文
set_locale("en")

# 获取当前语言
current_lang = get_current_locale()
```

### 2. 获取翻译文本

```python
# 获取节点显示名称
node_name = locale_manager.get_node_display_name("nanobanana apiAICG阿洋（文生图）")

# 获取控件标签
label = locale_manager.get_widget_label("api_key")

# 获取错误消息
error_msg = locale_manager.t("error_messages.api_key_empty")

# 获取格式化消息
formatted_msg = locale_manager.t("error_messages.image_conversion_failed", error="具体错误信息")
```

### 3. 在ComfyUI中使用

节点会自动根据当前语言设置显示相应的文本。用户可以通过以下方式切换语言：

```python
# 在ComfyUI控制台或脚本中
from comfyui_ayang_node import set_locale

# 切换到中文
set_locale("zh_CN")

# 切换到英文
set_locale("en")
```

## 翻译文件结构

翻译文件采用JSON格式，结构如下：

```json
{
    "node_display_names": {
        "节点内部名称": "显示名称"
    },
    "widget_labels": {
        "控件名称": "控件标签"
    },
    "error_messages": {
        "错误键": "错误消息"
    },
    "status_messages": {
        "状态键": "状态消息"
    },
    "prompts": {
        "提示键": "提示文本"
    }
}
```

## 添加新语言

1. 在 `locale/` 目录下创建新的语言目录（如 `ja` 表示日语）
2. 创建 `strings.json` 文件，包含所有翻译
3. 在 `LocaleManager` 类的 `load_translations` 方法中添加新语言

## 添加新翻译

1. 在相应的翻译文件中添加新的键值对
2. 在代码中使用 `locale_manager.t("category.key")` 获取翻译
3. 支持格式化字符串，使用 `{variable}` 语法

## 示例

运行 `locale_example.py` 查看完整的使用示例：

```bash
python locale_example.py
```

## 注意事项

- 默认语言为英文
- 如果找不到翻译，会返回键本身
- 支持嵌套键访问（如 `error_messages.api_key_empty`）
- 格式化字符串支持参数传递
- 语言切换会立即更新节点显示名称

## 技术实现

- 使用 `LocaleManager` 类管理多语言
- 支持动态加载翻译文件
- 提供便捷的翻译获取方法
- 集成到ComfyUI的节点系统中
