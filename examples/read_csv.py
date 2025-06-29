"""
Description: 
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 13:24
Version: 1.0
"""
#!/usr/bin/env python
"""
读取NPZ文件并显示其中所有数据的shape信息
"""

import numpy as np
import os
import sys


def read_npz_shapes(npz_path):
    """
    读取npz文件并打印所有数组的shape

    Args:
        npz_path: npz文件路径
    """
    # 处理Windows路径
    npz_path = os.path.normpath(npz_path)

    # 打印调试信息
    print(f"尝试读取文件: {npz_path}")
    print(f"当前工作目录: {os.getcwd()}")

    # 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"\n错误: 文件不存在 - {npz_path}")

        # 尝试其他可能的路径
        print("\n尝试查找文件...")

        # 尝试相对路径
        filename = os.path.basename(npz_path)
        if os.path.exists(filename):
            print(f"找到文件在当前目录: {filename}")
            npz_path = filename
        else:
            # 列出当前目录的npz文件
            npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]
            if npz_files:
                print(f"\n当前目录下的NPZ文件:")
                for f in npz_files:
                    print(f"  - {f}")
            else:
                print("当前目录下没有找到NPZ文件")
            return

    # 检查文件扩展名
    if not npz_path.endswith('.npz'):
        print(f"警告: 文件可能不是npz格式 - {npz_path}")

    try:
        # 加载npz文件
        data = np.load(npz_path)

        print(f"\n文件: {npz_path}")
        print("-" * 50)

        # 获取所有的key并排序
        keys = sorted(data.keys())

        if len(keys) == 0:
            print("文件中没有数据")
            return

        print(f"包含 {len(keys)} 个数组:\n")

        # 遍历并打印每个数组的信息
        for key in keys:
            arr = data[key]
            print(f"  '{key}':")
            print(f"    - Shape: {arr.shape}")
            print(f"    - Dtype: {arr.dtype}")
            print(f"    - Size: {arr.size} elements")
            print(f"    - Memory: {arr.nbytes / 1024 / 1024:.2f} MB")
            print()

        # 计算总大小
        total_size = sum(data[key].nbytes for key in keys)
        print("-" * 50)
        print(f"总内存占用: {total_size / 1024 / 1024:.2f} MB")

        # 关闭文件
        data.close()

    except Exception as e:
        print(f"读取文件时出错: {e}")


def main():
    # 默认文件路径 - 使用原始字符串避免转义问题
    default_path = r"Z:\yufengjiang\data\slice\002_S_0295_I40966_reg_slice_13.npz"

    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = default_path

        # 如果默认路径不存在，提示用户输入
        if not os.path.exists(default_path):
            print(f"默认路径不存在: {default_path}")
            print("\n请输入NPZ文件的完整路径（或拖拽文件到这里）:")
            user_input = input().strip()

            # 移除可能的引号
            if user_input.startswith('"') and user_input.endswith('"'):
                user_input = user_input[1:-1]

            npz_path = user_input

    # 读取并显示shape信息
    read_npz_shapes(npz_path)


if __name__ == "__main__":
    main()