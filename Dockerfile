# Use official Python image - updated to Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for lxml, nltk, and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy the rest of the project files
COPY . .

# Default port for Streamlit
EXPOSE 8501

# Command to run Streamlit app by default
CMD ["streamlit", "run", "demo.py", "--server.address=0.0.0.0"]
