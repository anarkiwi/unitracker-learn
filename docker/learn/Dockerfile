ARG BASE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime

FROM ${BASE} AS ml
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
COPY docker/learn/requirements.txt /tmp/
RUN python -c "import torch" 2>/dev/null \
    || pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /tmp/requirements.txt

FROM ml AS deps
COPY pyproject.toml /tmp/
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt

FROM deps AS test
WORKDIR /src
COPY . .
RUN pip install --no-deps -e .
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CMD ["sh", "-c", "black --check src tests && pylint src && pylint --disable=missing-function-docstring tests && pytest -n auto --cov=unitracker_learn --cov-fail-under=85"]
