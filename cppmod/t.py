import freeimage
import numpy
from pathlib import Path
import sys

ims = [freeimage.read(p).astype(numpy.float32) for p in ('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1508 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1528 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1548 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1608 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1637 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1704 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1714 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1728 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1748 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1808 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1828 bf_ffc.png')]
image_stack = numpy.dstack(ims)
mask = freeimage.read('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/non-vignette.png')
median = mask.astype(numpy.float32)
sys.path.insert(0, str(Path(__file__).parent))
from cppmod import _cppmod
_cppmod.image_stack_median(image_stack, mask, median)
