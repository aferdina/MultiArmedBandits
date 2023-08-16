FROM python:3.10-buster as builder

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.10-slim-buster as runtime

ENV VIRTUAL_ENV=/.venv \
    PATH="/.venv/bin:$PATH" \
    PYTHONPATH="/"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY multiarmedbandits ./multiarmedbandits

CMD /bin/bash