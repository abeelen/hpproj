FROM debian:testing-slim
MAINTAINER Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>
RUN apt-get update
RUN apt-get install -y tox
# Basic python v2 packages
RUN apt-get install -y python2.7 python-pip python-pytest-cov python-coverage python-subprocess32
RUN apt-get install -y python-numpy python-matplotlib python-healpy python-astropy python-photutils python-astroquery
RUN pip install pytest-mpl --no-deps
# Basic python v3 packages
RUN apt-get install -y python3.6 python3-pip python3-pytest-cov python3-coverage
RUN apt-get install -y python3-numpy python3-matplotlib python3-healpy python3-astropy python3-photutils python3-astroquery
RUN pip3 install pytest-mpl --no-deps
