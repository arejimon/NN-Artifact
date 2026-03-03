FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# System dependencies for SimpleITK/ITK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pymidas
COPY pymidas/ /app/pymidas/
RUN pip install --no-cache-dir /app/pymidas/

# Install artifactremoval package
COPY src/ /app/src/
COPY setup.py /app/
RUN pip install --no-cache-dir -e .

# Copy inference entrypoint
COPY inference.py /app/

# Copy model weights (baked into image)
COPY models/ /app/models/

ENTRYPOINT ["python", "inference.py"]
