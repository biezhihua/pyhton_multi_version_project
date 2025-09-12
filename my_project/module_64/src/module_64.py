import sys
import os
import sys
from pathlib import Path

main_64_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../main_64/src'))
if main_64_src not in sys.path:
    sys.path.insert(0, main_64_src)

shared_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from shared.utils.communication import SocketServer
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