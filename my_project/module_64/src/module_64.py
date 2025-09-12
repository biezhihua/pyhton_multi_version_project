# 64位模块实现
import sys
from pathlib import Path

# 添加共享目录到路径
shared_path = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(shared_path))

from communication import SocketServer
from shared.utils.logger import get_logger

logger = get_logger("module_64")

def message_handler(message):
    """处理收到的消息"""
    action = message.get("action")
    
    if action == "ping":
        return {"response": "pong", "module": "64-bit"}
    
    elif action == "process":
        data = message.get("data")
        # 处理数据
        result = f"Processed by 64-bit module: {data}"
        return {"result": result}
    
    else:
        return {"error": f"Unknown action: {action}"}

def main():
    logger.info("Starting 64-bit module...")
    
    # 启动通信服务器
    server = SocketServer('localhost', 9001, message_handler)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down 64-bit module...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()