# install cuda-toolkit to conda environment
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

# installing pytorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# installing dgl
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# installing transformers + accelerate + peft
pip install transformers==4.51.3 accelerate peft

# install DeepSpeed (optional - for 20-40% training speedup)
# Compatible version for transformers 4.51.3 + PyTorch 2.4.0
pip install deepspeed==0.15.4

# install ring_flash_attn + flash_attn
pip install ring-flash-attn
pip install flash-attn==2.7.1-post4 --no-build-isolation

# install other packages
pip install rich numpy cpgqls_client nest-asyncio coverage scikit-learn
pip install anthropic openai