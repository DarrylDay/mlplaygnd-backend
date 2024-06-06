# start by pulling the python image
FROM python:3.12

ENV DEBUG=True
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

EXPOSE 5000

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["server.py" ]