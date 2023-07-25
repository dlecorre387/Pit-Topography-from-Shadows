# Build from OSGeo docker image with gdal and ogr python APIs pre-installed
FROM osgeo/gdal:ubuntu-small-latest

# Set author for dockerfile
LABEL org.opencontainers.image.authors="Daniel Le Corre <dl387@kent.ac.uk / www.danlecorre.com>"

# Update package list and install python 3
RUN apt update && apt install --no-install-recommends -y    \
    python3-pip                                             \
    python3-dev                                             \
    git

# Install python packages
RUN pip3 --no-cache-dir install \
    numpy                       \
    matplotlib                  \
    python-math                 \
    scikit-learn                \
    scikit-image                \
    scipy                       \
    tqdm                        \
    notebook                    \
    jupyterlab                  \
    ipywidgets                  \
    ipykernel

# Copy scripts to the relevant folder in container
ADD scripts /app

# Expose port
EXPOSE 8888

# Run PITS with Jupyter notebook frontend
CMD [ "sh", "-c", "jupyter lab --ip 0.0.0.0 --port=8888 --allow-root --ServerApp.base_url=$PATH_PREFIX --ServerApp.password='' --ServerApp.token=''"]