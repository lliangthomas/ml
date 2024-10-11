# No PyTorch image because no NVIDIA GPU :(
FROM python:3.10-slim

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir fastapi uvicorn \
    pandas numpy matplotlib

WORKDIR /workspace
COPY . /workspace
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]