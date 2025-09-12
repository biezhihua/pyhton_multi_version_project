# 进程管理模块
import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path
from communication import SocketClient

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent.parent.parent
        
    def start_module(self, module_name, python_executable=None):
        """启动指定模块"""
        module_path = self.base_dir / module_name / "src" / f"{module_name}.py"
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module {module_name} not found at {module_path}")
        
        # 确定Python解释器
        if python_executable is None:
            python_executable = sys.executable
            if module_name == "module_32":
                # 尝试查找32位Python
                python_executable = self._find_32bit_python()
        
        # 启动进程
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_dir / "shared") + os.pathsep + env.get("PYTHONPATH", "")
        
        process = subprocess.Popen(
            [python_executable, str(module_path)],
            cwd=str(self.base_dir / module_name),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes[module_name] = process
        
        # 启动输出监控线程
        threading.Thread(
            target=self._monitor_output,
            args=(module_name, process),
            daemon=True
        ).start()
        
        return process
    
    def _find_32bit_python(self):
        """查找32位Python解释器"""
        # 这里可以根据实际情况调整查找逻辑
        possible_paths = [
            "C:/Python32/python.exe",
            "C:/Python32-32/python.exe",
            # 添加其他可能的路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise Exception("32-bit Python not found. Please specify the path manually.")
    
    def _monitor_output(self, module_name, process):
        """监控子进程输出"""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{module_name}] {output.strip()}")
        
        # 检查是否有错误输出
        error_output = process.stderr.read()
        if error_output:
            print(f"[{module_name} ERROR] {error_output}")
    
    def stop_module(self, module_name):
        """停止指定模块"""
        if module_name in self.processes:
            process = self.processes[module_name]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.processes[module_name]
    
    def stop_all(self):
        """停止所有模块"""
        for module_name in list(self.processes.keys()):
            self.stop_module(module_name)
    
    def is_running(self, module_name):
        """检查模块是否运行"""
        if module_name not in self.processes:
            return False
        return self.processes[module_name].poll() is None

# 单例模式
process_manager = ProcessManager()