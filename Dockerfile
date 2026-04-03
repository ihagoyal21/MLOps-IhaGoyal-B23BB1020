FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio \
    transformers datasets peft \
    wandb optuna matplotlib seaborn \
    adversarial-robustness-toolbox \
    scikit-learn tqdm

CMD ["bash"]