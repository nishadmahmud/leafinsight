FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by OpenCV and Python packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .

# Allow toggling between CPU and GPU PyTorch via build arguments (defaults to CPU)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX_URL}

# Install remaining dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
