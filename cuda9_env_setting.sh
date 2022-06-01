conda create -n unsup3d_cuda9 python=3.7.13
conda activate unsup3d_cuda9

pip install scikit-image matplotlib opencv-python moviepy pyyaml tensorboardX pytictoc
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install neural_renderer_pytorch

pip install --upgrade protobuf==3.20.0
pip install --upgrade pillow==8.3.2