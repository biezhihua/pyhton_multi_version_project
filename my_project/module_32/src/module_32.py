# 32位模块实现
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

logger = get_logger("module_32")

def message_handler(message):
    """处理收到的消息"""
    action = message.get("action")
    
    if action == "ping":
        return {"response": "pong", "module": "32-bit"}
    
    elif action == "process_32bit":
        data = message.get("data")
        # 这里可以调用32位特定的功能，如32位DLL
        result = f"Processed by 32-bit module: {data}"
        return {"result": result}
    
    else:
        return {"error": f"Unknown action: {action}"}

def main():
    logger.info("Starting 32-bit module...")
    
    # 启动通信服务器
    server = SocketServer('localhost', 9002, message_handler)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down 32-bit module...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()