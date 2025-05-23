FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies that might be needed for some Python packages
# and for profila (e.g., gdb needs to be available for profila setup initially)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set up Profila (installs its own gdb)
# This needs to be run after Python dependencies (including profila) are installed
# Pipe "yes" to the command to automatically accept the gdb download prompt
RUN echo "Y" | python -m profila setup

# Expose any ports if your application is a web server (not applicable here)

# Default command to run when the container starts
# This will execute your main script with Profila and output to stdout
# Adjust MAX_DEG and other parameters in src/main.py as needed for profiling runs.
CMD ["python", "-m", "profila", "annotate", "--", "src/main.py"] 