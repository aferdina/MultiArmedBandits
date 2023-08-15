FROM python:3.10
RUN pip install --upgrade pip
RUN pip install \
    pip install poetry
ADD pyproject.toml poetry.lock /code/
RUN poetry install

CMD /bin/bash