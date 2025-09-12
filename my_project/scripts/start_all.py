# 一键启动脚本
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# 添加 shared 到路径
shared_path = Path(__file__).parent.parent / "shared"
sys.path.insert(0, str(shared_path))

# 添加主工程到路径
main_path = Path(__file__).parent.parent / "main_64" / "src"
sys.path.insert(0, str(main_path))

from main import main

if __name__ == "__main__":
    sys.exit(main())