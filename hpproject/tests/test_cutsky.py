from .. import parse_args, combine_args
from .. import DEFAULT_npix, DEFAULT_coordframe, DEFAULT_pixsize

def test_parse_args_empty():
    try:
        args = parse_args(' '.split())
        assert False
    except SystemExit:
        assert True

def test_parse_args_defaults():
    args = parse_args('0.0 0.0 '.split())
    (npix, pixsize, coordframe, ctype, maps) = combine_args(args, {})

    assert(npix == DEFAULT_npix)
    assert(coordframe == DEFAULT_coordframe)
    assert(pixsize == DEFAULT_pixsize)

def test_combine_args_radius():
    args = parse_args('0.0 0.0 --pixsize 1 --radius 60'.split())
    (npix, pixsize, coordframe, ctype, maps) = combine_args(args, {})
    assert(pixsize == 1)
    assert(npix == 3600)
