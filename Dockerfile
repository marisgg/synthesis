ARG paynt_base=randriu/paynt:ci

# Pull paynt
FROM $paynt_base

WORKDIR /opt/learning

# Install PyTorch and Jax with CUDA support.
RUN pip install torch==2.4.* "jax[cuda12]"

# Additional dependencies.
RUN pip install ipykernel joblib tensorboard==2.15.* einops==0.7.* gym==0.22.* pygame==2.5.* tqdm

RUN apt-get update && apt-get install -y curl

# install VS Code (code-server)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# install VS Code extensions
RUN code-server --install-extension ms-python.python \
                --install-extension ms-toolsai.jupyter
