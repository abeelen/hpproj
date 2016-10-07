import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

opts = dict(name="hpproj",
            author='Alexandre Beelen',
            author_email='alexandre.beelen@ias.u-psud.fr',
            maintainer="Marian Douspis",
            maintainer_email="marian.douspis@ias.u-psud.fr",
            description='Projection of Healpix maps onto a planar grid',
            long_description='Projection of Healpix maps onto a planar grid using wcs headers',
            url='https://git.ias.u-psud.fr/abeelen/hpproj',
            download_url='https://git.ias.u-psud.fr/abeelen/hpproj/repository/archive.tar.gz?0.2',
            license='LGPL-3.0+',
            classifiers=['Topic :: Scientific/Engineering :: Astronomy',
                         'Intended Audience :: Science/Research',
                         'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)'],
            version='0.2',
            packages=['hpproj'],
            package_dir={'hpproj'  : 'hpproj'},
            entry_points = {
                'console_scripts': [
                    'cutsky = hpproj.cutsky:main'] },
            setup_requires=['pytest-runner'],
            tests_require=['pytest'],

            requires=['numpy', 'matplotlib', 'healpy', 'astropy',
                      'photutils' ],
)


if __name__ == '__main__':
    setup(**opts)
