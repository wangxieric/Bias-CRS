# Installation Logs

## Errors:

> zipfile.BadZipFile: Bad CRC-32 for file 'torch/lib/libtorch_cpu.so'
- `--no-cache` parameter should be considered in the first command:
> pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache