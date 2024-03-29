master
======


Added
-----

Fixed
-----


Version 0.9.4
=============

Fixed
-----
* Updated compatibility with numpy and matplotlib


Version 0.9.3
=============

Added
-----
* `Cutsky` and `build_hpmap` can handle a tuple (np.array, fits.Header) as input instead of filename

Fixed
-----
* logger are not configured in the module 

Version 0.9.2
=============

Added
-----
* any header attribute can be changed in the config file

Fixed
-----
* `doContour` and `doCut` options changed to lowercase


Version 0.9.1
==============

Added
-----
* `_cut_wcs` method to `CutSky`
* `CutSky` now use SkyCoord internally

Fixed
-----
* Updated test matrix, added py3.7, special care of py3.5
* Fixed tests failing for photutils 0.6 new API

Version 0.9.0
=============

Added
-----
* `apertures` key in the map argument can be configured as float or list of float in arcminute
* `--votable` option require aperture or aperture list in arcminute

Version 0.8.1
=============

Fixed
-----
* New test matrix, now depends on astropy > 2.0 (from photutils)

Version 0.8.0
=============

Added
-----
* Profile routine
* `hp_project` use shape_out instead of npix
* Add `hp_stack` to perform stacking on healpix map
* Add `hp_profile` to extract profile at given SkyCoord
* Add `hp_photometry` to perform aperture photometry on list of SkyCoord

Fixed
-----
* Refactoring : split wcs_helper from hp_helper
* Refactoring : Remove npix arguments from internal functions

Version 0.7.4
=============

Fixed
-----
* doc build process
* docstring glitch on decorator
* bug in function decorator
* documentation on internal calls & limitations
* bug when reading nested healpix maps

Version 0.7.0
=============

Added
-----
* Unit tests on the plotting functions thanks to `pytest-mpl`
* SonarQube integration with travis
* decorator changed :
    - hphdu_to_* functions merged into hp_to, function accept either a :class:`astropy.io.fits.ImageHDU` or array_like and :class:`astropy.fits.header.Header`
    - *_lonlat	 functions merged into build_wcs*, function accept either a :class:`astropy.coordinate.SkyCoord` or 2 floats and a keyword : `lon, lat, src_frame='EQUATORIAL'`

Fixed
-----
* Moved tests scripts into package root
* Refactoring of some function thanks to SonarQube
* :func:`~hpproj.hp_helpers.hp_project` now always returns an :class:`astropy.io.fits.PrimaryHDU`

Version 0.6.1
=============

Added
-----
* Documentation for visualization function with examples

Fixed
-----
* Unit tests (travis -py3.4 +py3.6)
* Documentation links

Version 0.6.0
=============

Added
-----
* Visualization function (`mollview`, `carview`, `merview`, `coeview`, `bonview`, `pcoview`, `tscview`, `orthview`)

Version 0.5.0
=============

Added
-----
* ExtendedInterpolation for configparser for py3, only for filename in py2
* ``maps_selection`` option in the cut_ functions to allow sub_sample selection
* `--xml` will now overwrite pre-existing file

Fixed
-----
* `build_wcs*` nows put the projection center exactly at the center of the map (0.5 pixel off before)

Version 0.4.0
=============

Added
-----
* doCut is now implicit to True in the config file
* cutsky() now can use both old and new definition of maps
* rtd pages

Fixed
-----
* Important refactoring of the code

Version 0.3.3
=============

Added
-----
* Full python 2 and python 3 compatibility
* Basic README

Version 0.3
===========

Added
-----
* output directory option
* Full CI test including cutsky()

Fixed
-----
* wcsaxes in requirements
* small license in all python file
* Minor bugs

Version 0.2.1
=============

Fixed
-----
* Major Bug in uncovered code
* Typo

Version 0.2
===========

Added
-----
* Ci
* verbosity options
* output options

Fixed
-----
* several small bug fixes

Version 0.1
===========

Added
-----
* Initial version, basic output capabilities, no test sui
