#!/usr/bin/env python3
"""
多语言支持使用示例
"""

# 导入模块
from __init__ import set_locale, get_available_locales, get_current_locale, locale_manager

def main():
    print("=== AICG Ayang Node 多语言支持示例 ===\n")
    
    # 显示可用的语言
    available_locales = get_available_locales()
    print(f"可用语言: {available_locales}")
    print(f"当前语言: {get_current_locale()}\n")
    
    # 测试英文
    print("--- 英文模式 ---")
    set_locale("en")
    print(f"当前语言: {get_current_locale()}")
    print(f"文生图节点: {locale_manager.get_node_display_name('nanobanana apiAICG阿洋（文生图）')}")
    print(f"图生图节点: {locale_manager.get_node_display_name('nanobanana apiAICG阿洋（图生图）')}")
    print(f"尺寸调整节点: {locale_manager.get_node_display_name('nanobanana 图生图尺寸调整')}")
    print(f"API密钥标签: {locale_manager.get_widget_label('api_key')}")
    print(f"提示词标签: {locale_manager.get_widget_label('prompt')}")
    print(f"错误消息: {locale_manager.t('error_messages.api_key_empty')}")
    print(f"成功消息: {locale_manager.t('status_messages.success_text_to_image', count=1, key_num=1, attempt=1, total=1)}\n")
    
    # 测试中文
    print("--- 中文模式 ---")
    set_locale("zh_CN")
    print(f"当前语言: {get_current_locale()}")
    print(f"文生图节点: {locale_manager.get_node_display_name('nanobanana apiAICG阿洋（文生图）')}")
    print(f"图生图节点: {locale_manager.get_node_display_name('nanobanana apiAICG阿洋（图生图）')}")
    print(f"尺寸调整节点: {locale_manager.get_node_display_name('nanobanana 图生图尺寸调整')}")
    print(f"API密钥标签: {locale_manager.get_widget_label('api_key')}")
    print(f"提示词标签: {locale_manager.get_widget_label('prompt')}")
    print(f"错误消息: {locale_manager.t('error_messages.api_key_empty')}")
    print(f"成功消息: {locale_manager.t('status_messages.success_text_to_image', count=1, key_num=1, attempt=1, total=1)}\n")
    
    # 测试格式化字符串
    print("--- 格式化字符串测试 ---")
    set_locale("en")
    print(f"英文错误: {locale_manager.t('error_messages.image_conversion_failed', error='Test error')}")
    set_locale("zh_CN")
    print(f"中文错误: {locale_manager.t('error_messages.image_conversion_failed', error='测试错误')}")

if __name__ == "__main__":
    main()
