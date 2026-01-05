# **Progressively Label Enhancement for Large Language Model Alignment**





**ICML 2025 Accepted Paper**


📄 **Paper**: https://arxiv.org/abs/2408.02599

📍 **Conference**: International Conference on Machine Learning (ICML), 2025



------



## 🚧 Code Release Notice

本仓库提供 PLE 在 OpenRLHF 框架下的代码实现。  
下面给出从零开始运行默认实验脚本的完整步骤。

```bash
# 克隆仓库
git clone https://github.com/palm-biaoliu/PLE.git
cd PLE

# 创建并激活虚拟环境（可选）
conda create -n ple python=3.10
conda activate ple

# 安装依赖
pip install -r requirements.txt
# 或者（如果你使用的是 pyproject）
# pip install -e .

# 运行默认脚本
cd examples/scripts
bash run_ple.sh
```








