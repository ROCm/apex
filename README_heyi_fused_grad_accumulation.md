docker pull rocm/pytorch:latest

docker run -it --network=host -v /home/yigex/huggingface/heyi:/workspace  --device=/dev/kfd --device=/dev/dri --group-add video  --security-opt seccomp=unconfined --ipc=host --name heyi_apex_wx  rocm/pytorch:latest

mkdir heyi && cd heyi

git clone -b heyi_fused_grad_accumulation https://github.com/eliotwang/apex

cd apex

python setup.py install --cpp_ext --cuda_ext

python tests/L0/run_transformer/test_weight_grad.py
or
python tests/L0/run_test.py --include run_transformer
