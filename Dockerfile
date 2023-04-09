FROM python:3.10.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE $PORT
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app
