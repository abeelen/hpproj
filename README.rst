HealPixProjection
=================

|pypi| |license| |wheels| |format| |pyversions| |rtd|

HealPixProjection is a project to allow easy and efficient projection of healpix maps onto planar grids. It can be used as a standalone program ``cutsky``

.. code:: bash

    $ cutsky 0.0 0.0 --mapfilenames HFI_SkyMap_857_2048_R2.00_full.fits

or as a python function

.. code:: python

    from hpproj import cutsky
    result = cutsky([0.0, 0.0], maps={'Planck 857': {'filename': 'HFI_SkyMap_857_2048_R2.00_full.fits'}} )

or as a python class, for optimization when producting several cuts

.. code:: python

    from hpproj import CutSky, to_coord
    cutsky = CutSky({'Planck 857': {'filename': 'HFI_SkyMap_857_2048_R2.00_full.fits'}})
    result = cutsky.cut_fits(to_coord([0., 0.])

.. note::

    For science applications requiring high photometric accuracy, we recommend the drizzlib software developed by CADE, which uses a flux-conserving drizzling method to reproject data between HEALPix and local WCS. Drizzlib is available as standalone software (IDL python) here: http://cade.irap.omp.eu/dokuwiki/doku.php?id=software . An online interface, drizzweb, is available here: http://drizzweb.irap.omp.eu/ .

Features
--------

- Galactic and equatorial system supported
- All projection system from ``wcslib``
- Project several healpix maps at once, efficiently !
- Output in ``fits``, ``png`` or ``votable`` for the central point source photometry

Installation
------------

Install ``hpproj`` using pip :

.. code:: bash

    $ pip install hpproj

or by running setuptools on `source <https://git.ias.u-psud.fr/abeelen/hpproj/tree/master>`_


.. code:: bash

    $ python setup.py install

Contribute
----------

- `Issues Tracker <https://git.ias.u-psud.fr/abeelen/hpproj/issues>`_
- `Source Code <https://git.ias.u-psud.fr/abeelen/hpproj/tree/master>`_

Support
-------

If you are having issues, please let us know.

License
-------

This project is licensed under the LGPL+3.0 license.

|build-gitlabCI| |build-travis| |codeclimate| |codehealth| |coverage| |sonarqube|


.. |pypi| image:: https://img.shields.io/pypi/v/hpproj.svg?maxAge=2592000
    :alt: Latest Version
    :target: https://pypi.python.org/pypi/hpproj


.. |license| image:: https://img.shields.io/pypi/l/hpproj.svg?maxAge=2592000
    :alt: License


.. |wheels| image:: https://img.shields.io/pypi/wheel/hpproj.svg?maxAge=2592000
   :alt: Wheels


.. |format| image:: https://img.shields.io/pypi/format/hpproj.svg?maxAge=2592000
   :alt: Format


.. |pyversions| image:: https://img.shields.io/pypi/pyversions/hpproj.svg?maxAge=25920001;5002;0c
   :alt: pyversions


.. |build-gitlabCI| image:: https://git.ias.u-psud.fr/abeelen/hpproj/badges/master/build.svg
    :alt: Gitlab-CI Master Build
    :target: https://git.ias.u-psud.fr/abeelen/hpproj/builds


.. |build-travis| image:: https://travis-ci.org/abeelen/hpproj.svg?branch=master
    :alt: Travis Master Build
    :target: https://travis-ci.org/abeelen/hpproj


.. |codeclimate| image:: https://codeclimate.com/github/abeelen/hpproj/badges/gpa.svg
   :alt: Code Climate
   :target: https://codeclimate.com/github/abeelen/hpproj


.. |codehealth| image:: https://landscape.io/github/abeelen/hpproj/master/landscape.svg?style=flat
   :alt: Code Health
   :target: https://landscape.io/github/abeelen/hpproj/master


.. |sonarqube| image:: https://sonarcloud.io/api/project_badges/measure?project=hpproj&metric=alert_status
   :alt: SonarQube
   :target: https://sonarcloud.io/dashboard/index/hpproj



.. |coverage| image:: https://git.ias.u-psud.fr/abeelen/hpproj/badges/master/coverage.svg
    :alt: Master Coverage


.. |rtd| image:: https://readthedocs.org/projects/hpproj/badge/?version=latest
    :alt: Read the doc
    :target: http://hpproj.readthedocs.io/
