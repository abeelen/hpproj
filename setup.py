import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

opts = dict(name="hpproj",
            maintainer="Marian Douspis",
            maintainer_email="marian.douspis@ias.u-psud.fr",
            description='Projection of Healpix maps onto a planar grid',
            long_description='Projection of Healpix maps onto a planar grid using wcs headers',
#            url=URL,
            #            download_url=DOWNLOAD_URL,
            license='LGPL-3.0+',
            classifiers=['Topic :: Scientific/Engineering :: Astronomy',
                         'Intended Audience :: Science/Research',
                         'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)'],
            author='Alexandre Beelen',
            author_email='alexandre.beelen@ias.u-psud.fr',
#            platforms=PLATFORMS,
            version='0.1',
            packages=['hpproj'],
            package_dir={'hpproj'  : 'hpproj'},
            entry_points = {
                'console_scripts': [
                    'cutsky = hpproj.cutsky:main'] },
            #            package_data=PACKAGE_DATA,
            #            requires=REQUIRES,
)


if __name__ == '__main__':
    setup(**opts)
