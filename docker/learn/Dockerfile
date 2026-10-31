FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime AS ml
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY docker/learn/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM ml AS deps
COPY pyproject.toml /tmp/
RUN python -c "import tomllib; p = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['tokenize']))" > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt
