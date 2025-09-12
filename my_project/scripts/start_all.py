# 一键启动脚本
#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加主工程到路径
main_path = Path(__file__).parent.parent / "main_64" / "src"
sys.path.insert(0, str(main_path))

from main import main

if __name__ == "__main__":
    sys.exit(main())