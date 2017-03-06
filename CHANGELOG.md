master
======

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
