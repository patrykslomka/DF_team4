# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for lxml, nltk, and other packages
RUN apt-get update && \
    apt-get install -y build-essential gcc libxml2-dev libxslt1-dev python3-dev libffi-dev libssl-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download NLTK data (optional, can be commented out if not needed)
RUN python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Copy the rest of the project files
COPY . .

# Default command: open bash shell (user can run any script)
CMD ["bash"]
