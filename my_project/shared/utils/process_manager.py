# 进程管理模块
import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path
from shared.utils.communication import SocketClient
import yaml

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent.parent.parent
        self.config = self._load_config()

    def _load_config(self):
        """加载主工程配置文件"""
        config_path = self.base_dir / "main_64" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def start_module(self, module_name, python_executable=None):
        """启动指定模块"""
        module_path = self.base_dir / module_name / "src" / f"{module_name}.py"

        if not module_path.exists():
            raise FileNotFoundError(f"Module {module_name} not found at {module_path}")

        # 确定Python解释器
        if python_executable is None:
            if module_name == "module_64":
                python_path = self.config.get('modules', {}).get('64_bit', {}).get('python_path')
                if python_path and os.path.exists(python_path):
                    python_executable = python_path
                else:
                    python_executable = self._find_64bit_python()

            if module_name == "module_32":
                python_path = self.config.get('modules', {}).get('32_bit', {}).get('python_path')
                if python_path and os.path.exists(python_path):
                    python_executable = python_path
                else:
                    python_executable = self._find_32bit_python()

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

        threading.Thread(
            target=self._monitor_output,
            args=(module_name, process),
            daemon=True
        ).start()

        return process
    
    def _find_64bit_python(self):
        possible_paths = [
            "C:/Python64/python.exe",
            "C:/Python64-64/python.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise Exception("64-bit Python not found. Please specify the path manually.")
    
    def _find_32bit_python(self):
        possible_paths = [
            "C:/Python32/python.exe",
            "C:/Python32-32/python.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise Exception("32-bit Python not found. Please specify the path manually.")
    
    def _monitor_output(self, module_name, process):
        for line in process.stdout:
            print(f"[{module_name} INFO] {line.strip()}")
        for line in process.stderr:
            print(f"[{module_name} ERROR] {line.strip()}")
    
    def stop_module(self, module_name):
        process = self.processes.get(module_name)
        if process:
            process.terminate()
            process.wait()
            del self.processes[module_name]
    
    def stop_all(self):
        for module_name in list(self.processes.keys()):
            self.stop_module(module_name)

    def is_running(self, module_name):
        """检查模块是否运行"""
        if module_name not in self.processes:
            return False
        return self.processes[module_name].poll() is None