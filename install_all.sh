# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install "vllm>=0.5.1,<0.9" -U
pip install "lmdeploy>=0.5" -U --no-deps
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
pip install 'ms-swift[all]'==3.5.0 -U
pip install timm -U
pip install deepspeed -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit -U
pip install "deepspeed<0.17" # https://github.com/modelscope/ms-swift/issues/4586
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
