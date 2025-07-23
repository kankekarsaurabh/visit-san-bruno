# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Command to run your app using Uvicorn when the container starts
CMD ["uvicorn", "sanbruno:app", "--host", "0.0.0.0", "--port", "8000"]