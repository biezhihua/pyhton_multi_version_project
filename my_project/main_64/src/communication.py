# 通信模块
import socket
import json
import struct
import threading
from shared.utils.logger import get_logger

logger = get_logger("communication")

class SocketClient:
    def __init__(self, host='localhost', port=9000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def send_message(self, message):
        """发送消息"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # 将消息转换为JSON并编码
            message_json = json.dumps(message).encode('utf-8')
            # 发送消息长度
            self.socket.sendall(struct.pack('>I', len(message_json)))
            # 发送消息内容
            self.socket.sendall(message_json)
            
            # 接收响应长度
            raw_len = self._recv_all(4)
            if not raw_len:
                return None
            resp_len = struct.unpack('>I', raw_len)[0]
            
            # 接收响应内容
            response = self._recv_all(resp_len)
            if response:
                return json.loads(response.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Send message failed: {e}")
            self.connected = False
            return None
    
    def _recv_all(self, n):
        """接收指定长度的数据"""
        data = b''
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
        self.connected = False

class SocketServer:
    def __init__(self, host='localhost', port=9000, message_handler=None):
        self.host = host
        self.port = port
        self.message_handler = message_handler
        self.socket = None
        self.running = False
        
    def start(self):
        """启动服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        logger.info(f"Server started on {self.host}:{self.port}")
        
        # 接受连接
        while self.running:
            try:
                conn, addr = self.socket.accept()
                logger.info(f"Connection from {addr}")
                # 为每个连接创建线程
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                ).start()
            except Exception as e:
                if self.running:
                    logger.error(f"Accept connection failed: {e}")
    
    def _handle_client(self, conn, addr):
        """处理客户端连接"""
        try:
            while True:
                # 接收消息长度
                raw_len = self._recv_all(conn, 4)
                if not raw_len:
                    break
                msg_len = struct.unpack('>I', raw_len)[0]
                
                # 接收消息内容
                data = self._recv_all(conn, msg_len)
                if not data:
                    break
                
                # 处理消息
                message = json.loads(data.decode('utf-8'))
                if self.message_handler:
                    response = self.message_handler(message)
                else:
                    response = {"status": "ok"}
                
                # 发送响应
                response_json = json.dumps(response).encode('utf-8')
                conn.sendall(struct.pack('>I', len(response_json)))
                conn.sendall(response_json)
                
        except Exception as e:
            logger.error(f"Handle client {addr} failed: {e}")
        finally:
            conn.close()
            logger.info(f"Connection from {addr} closed")
    
    def _recv_all(self, conn, n):
        """从连接接收指定长度的数据"""
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info("Server stopped")