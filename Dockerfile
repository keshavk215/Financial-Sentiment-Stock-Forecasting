# 1. Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .

# The CPU-only version of torch.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Copy all project files into the container
COPY . .

# 5. Expose the port the app will run on
EXPOSE 5000

# 6. Run the application
# Using Gunicorn, a production-ready web server, instead of Flask's built-in dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]

