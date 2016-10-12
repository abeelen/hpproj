FROM debian:stretch
MAINTAINER Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>
RUN apt-get update
RUN apt-get install -y tox
# Basic python v2 packages
RUN apt-get install -y python2.7
RUN apt-get install -y python-pip python-pytest-cov python-coverage
RUN apt-get install -y python-numpy python-matplotlib
RUN apt-get install -y python-astropy python-photutils python-healpy python-wcsaxes
# Basic python v3 packages
RUN apt-get install -y python3.5
RUN apt-get install -y python3-pip python3-pytest-cov python3-coverage
RUN apt-get install -y python3-numpy python3-matplotlib
RUN apt-get install -y python3-astropy python3-photutils python3-healpy python3-wcsaxes
