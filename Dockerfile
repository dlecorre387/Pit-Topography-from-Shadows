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

ADD MArtian_Pit_Shadow_extraction /app
ADD data /data

RUN mkdir /data/output/

WORKDIR /app