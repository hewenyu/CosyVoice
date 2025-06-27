#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from modelscope import snapshot_download

def download_if_not_exists(model_id, local_dir):
    """
    如果本地目录不存在，则从ModelScope下载模型
    
    Args:
        model_id: ModelScope模型ID
        local_dir: 本地保存目录
    """
    local_path = Path(local_dir)
    
    if local_path.exists() and any(local_path.iterdir()):
        print(f"模型 {model_id} 已存在于 {local_dir}，跳过下载")
        return True
    
    print(f"开始下载模型 {model_id} 到 {local_dir}")
    try:
        snapshot_download(model_id, local_dir=local_dir)
        print(f"模型 {model_id} 下载完成")
        return True
    except Exception as e:
        print(f"下载模型 {model_id} 失败: {str(e)}")
        return False

def main():
    # 确保目录存在
    os.makedirs("pretrained_models", exist_ok=True)
    
    # 模型列表
    models = [
        ('iic/CosyVoice2-0.5B', 'pretrained_models/CosyVoice2-0.5B'),
        ('iic/CosyVoice-300M', 'pretrained_models/CosyVoice-300M'),
        ('iic/CosyVoice-300M-SFT', 'pretrained_models/CosyVoice-300M-SFT'),
        ('iic/CosyVoice-300M-Instruct', 'pretrained_models/CosyVoice-300M-Instruct'),
        ('iic/CosyVoice-ttsfrd', 'pretrained_models/CosyVoice-ttsfrd'),
    ]
    
    success = True
    for model_id, local_dir in models:
        if not download_if_not_exists(model_id, local_dir):
            success = False
    
    if not success:
        print("有模型下载失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()