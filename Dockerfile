# Step 1: Use an official, stable Python runtime as our baseline environment
FROM python:3.11-slim

# Step 2: Set the internal folder inside the container where our app will live
WORKDIR /app

# Step 3: Install basic system utilities that backend math packages occasionally need
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy over just the environment blueprint first (optimizes Docker build speed)
COPY requirements.txt .

# Step 5: Install all the bioinformatics and AI packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your clean repository assets into the container
COPY . .

# Step 7: Tell the container to open up port 8501 (the standard Streamlit web gateway)
EXPOSE 8501

# Step 8: The exact command to boot up your orchestrator automatically when the container starts
ENTRYPOINT ["streamlit", "run", "ultimate_agent.py", "--server.port=8501", "--server.address=0.0.0.0"]