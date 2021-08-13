# git submodule add https://github.com/ROCmSoftwarePlatform/hipify-torch third_party/hipify-torch
rm -rf apex/contrib/csrc/groupbn/*_hip.h
PYTHONDONTWRITEBYTECODE=1 python3 third_party/hipify-torch/hipify_cli.py --project-directory apex/contrib/csrc/groupbn