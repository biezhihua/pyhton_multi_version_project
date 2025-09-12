#!/bin/bash
# 安装所有 requirements
#!/bin/bash

# 安装主工程依赖
echo "Installing main project requirements..."
pip install -r ../main_64/requirements.txt

# 安装64位模块依赖
echo "Installing 64-bit module requirements..."
pip install -r ../module_64/requirements.txt

# 安装32位模块依赖（使用32位Python）
echo "Installing 32-bit module requirements..."
# 这里需要指定32位Python的路径
/path/to/python32 -m pip install -r ../module_32/requirements.txt

echo "All requirements installed successfully."