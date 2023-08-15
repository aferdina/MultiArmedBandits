FROM python:3.10
RUN pip install --upgrade pip
RUN pip install \
    pip install poetry
RUN poetry config virtualenvs.create false \
    poetry install

CMD /bin/bash