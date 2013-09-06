import itertools

_shapes = ('rectangle', 'spline', 'circle', 'polygon', 'freehand', 'line', 'spoke')
_colormap = ('red', 'green', 'yellow', 'blue')

def _read_shape(line):
    shape,colorNum = line.split()
    return shape.lower(),_colormap[int(colorNum)]

def _read_numeric_cmd(line):
    return [s.strip(), line.strip().split())]

def _is_shape(line):
    chks = [line.lower().startswith(shape) for shape in _shapes]
    return any(chks)

def _is_eof(line):
    return line.lower().startswith('roiapproved')

def _shapes(f):
    shapes = {}
    shape = None
    for line in f:
        if _is_shape(line):
            shape = _read_shape(line)
            shapes[shape] = []
        elif _is_eof(line):
            pass # Ignore the EOF
        else:
            shapes[shape].append(line)
    return shapes

def _images(shapeData):
    images = {}
    image = None
    for line in shapeData:
        if _is_image(line):
            image = _read_numeric_cmd(line)
        else:
            images[image].append(line)
    return images

def _read_pt(line):
    return [float(d) for d in line.strip().split()]

def _pts(imageData):
    first,rest = imageData[0],imageData[1:]
    _,nPts = _read_numeric_cmd(first)
    dataPts = [_read_pt(pt) for pt in rest[:nPts]]
    return dataPts

def read_ros(roifile):
    with open(roifile) as f:
        _,version = f.readline().split()
        dims = map(int, f.readline().split()[1:])
       
        ndims = dims[4] if length(dims == 4) and dims[3] == 1 else dims[3]

         
        for shape,shapeData in _shapes(f).iteritems():
            for image,imageData in _images(shapeData).iteritems():
                pts = _pts(imageData)


