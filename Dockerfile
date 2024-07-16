# Use the official Python image.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container.
COPY . .

# Expose the port the app runs on.
EXPOSE 8888

# Run the command to start the FastAPI server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
