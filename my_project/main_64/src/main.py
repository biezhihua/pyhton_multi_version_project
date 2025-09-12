# 主程序入口

import signal
import sys
from pathlib import Path

# 添加共享目录到路径
shared_path = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))

from shared.utils.process_manager import ProcessManager
from shared.utils.communication import SocketClient
from shared.utils.logger import get_logger

logger = get_logger("main")

process_manager = ProcessManager()


def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info("Shutting down...")
    process_manager.stop_all()
    sys.exit(0)


def main():
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting main application...")

    try:
        # 启动64位模块
        logger.info("Starting 64-bit module...")
        process_manager.start_module("module_64")

        # 启动32位模块
        logger.info("Starting 32-bit module...")
        # 如果需要指定32位Python路径，可以在这里指定
        # process_manager.start_module("module_32", "C:/Python32/python.exe")
        process_manager.start_module("module_32")

        # 等待模块启动
        import time

        time.sleep(2)

        # 检查模块状态
        if not process_manager.is_running("module_64"):
            logger.error("64-bit module failed to start")
            return 1

        if not process_manager.is_running("module_32"):
            logger.error("32-bit module failed to start")
            return 1

        logger.info("All modules started successfully")

        # 示例：与模块通信
        client = SocketClient("localhost", 9001)  # 假设64位模块监听9001
        response = client.send_message({"action": "ping"})
        if response:
            logger.info(f"Response from 64-bit module: {response}")

        # 主循环
        while True:
            # 这里可以添加主程序逻辑
            time.sleep(1)

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        process_manager.stop_all()

    return 0


if __name__ == "__main__":
    sys.exit(main())
