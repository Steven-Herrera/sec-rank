FROM python:3.10.12-buster

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

# Install dependencies
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt