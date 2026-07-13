# g7_openarm_vla

```bash
git clone --recurse-submodules https://github.com/umeow0716/g7_openarm_vla.git
cd g7_openarm_vla
```

```bash
uv sync --all-packages

# If unitree_sdk2py install failed, To setup cyclonedds as unitree_sdk2py README.md (https://github.com/unitreerobotics/unitree_sdk2_python)
# CYCLONEDDS_HOME=~/cyclonedds/install uv sync --all-packages

uv run scripts/run_all.py
```
