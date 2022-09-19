FROM osgeo/gdal:ubuntu-small-latest

RUN apt update && apt install --no-install-recommends -y    \
    python3-pip

RUN pip3 --no-cache-dir install \
    numpy                       \
    python-math                 \
    scikit-learn                \
    scikit-image                \
    scipy                       \
    tqdm

ADD scripts /app
ADD data /data

WORKDIR /app