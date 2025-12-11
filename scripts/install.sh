# install cuda-toolkit to conda environment
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

# installing pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# installing dgl
# pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# install pytorch-geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# installing transformers + accelerate + peft
pip install transformers accelerate peft deepspeed

pip install ring-flash-attn
pip install flash-attn==2.7.2.post1 --no-build-isolation

# install other packages
pip install rich numpy pandas cpgqls_client nest-asyncio coverage scikit-learn anthropic openai GitPython