# backup_script.py

import os
import shutil
import sys

# --- 配置区 ---
# 你可以在这里修改备份文件夹的名称和目标文件类型
DESTINATION_FOLDER_NAME = "code_backup_1_6_TIME"
TARGET_EXTENSIONS = ('.py', '.txt')


# --- 配置区结束 ---

def create_backup():
    """
    主函数，执行备份逻辑。
    """
    try:
        # 1. 获取脚本所在的当前目录，这是我们的源目录
        # os.getcwd() 获取的是 "Current Working Directory"，即你从哪个目录执行的脚本
        # 为了确保我们总是处理脚本文件所在的目录，使用 __file__ 会更健壮
        source_directory = os.path.dirname(os.path.abspath(__file__))

        # 2. 构建目标备份文件夹的完整路径
        destination_directory = os.path.join(source_directory, DESTINATION_FOLDER_NAME)

        # 3. 检查并创建备份文件夹
        # os.makedirs() 可以创建多级目录
        # exist_ok=True 参数意味着如果文件夹已经存在，它不会抛出错误，这让脚本可以重复运行
        print(f"准备备份到文件夹: {destination_directory}")
        os.makedirs(destination_directory, exist_ok=True)

        # 获取脚本自身的文件名，以便在复制时可以跳过（可选）
        script_filename = os.path.basename(__file__)

        # 4. 遍历源目录中的所有项目
        copied_files_count = 0
        for item_name in os.listdir(source_directory):
            # 构建项目的完整路径
            item_path = os.path.join(source_directory, item_name)

            # 5. 筛选符合条件的文件
            #    - 必须是文件 (os.path.isfile)
            #    - 文件名必须以指定扩展名结尾 (item_name.endswith)
            #    - (可选) 避免复制备份文件夹本身的内容或脚本自身
            if os.path.isfile(item_path) and item_name.endswith(TARGET_EXTENSIONS):
                # 如果你想避免备份脚本自身，可以取消下面这行注释
                # if item_name == script_filename:
                #     continue

                # 6. 执行复制操作
                print(f"  -> 正在复制: {item_name}")

                # 使用 shutil.copy2 而不是 shutil.copy
                # copy2 会同时复制文件的元数据（如修改时间、创建时间），这对于备份更有意义
                shutil.copy2(item_path, destination_directory)
                copied_files_count += 1

        print("-" * 30)
        if copied_files_count > 0:
            print(f"备份完成！总共复制了 {copied_files_count} 个文件。")
        else:
            print("在当前目录下没有找到符合条件的 .py 或 .txt 文件。")

    except Exception as e:
        print(f"\n发生错误！备份中断。")
        print(f"错误详情: {e}")
        # 退出脚本并返回一个非零状态码，表示执行失败
        sys.exit(1)


if __name__ == "__main__":
    create_backup()

