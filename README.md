# E2FIF
Simple but stronge baseline for binarized super-resolution networks (BSRNs).

This repo is modified by [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch).

## Training

To train and reproduce the results of the paper, just run "train.sh/train_rcan.sh/train_rdn.sh".

Modify the params in "train.sh", like "model", "save", "binary_model" and so on. Then, 
'''shell
sh train.sh
'''


## Test your model on a mobile devices

By [Bolt](https://github.com/huawei-noah/bolt).

Step 1. Prepare your binarized onnx model. It should be noted that the BN layer will be fused with Conv when pytorch is converted to onnx, which may destroy the binarized conv layers.

Step 2. See the [Start pape for Bolt](https://github.com/huawei-noah/bolt) and select your *target platform* and *build platform*, like:
'''shell
./install.sh --target=android-aarch64 --gpu
'''

Step 3. Copy the compiled X2bolt, benchmark, and your onnx models to your phone by adb.

Step 4. Test the real latency of your models.
