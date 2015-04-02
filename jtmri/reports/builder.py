#!/usr/bin/env python
import jinja2
import matplotlib as mpl
import pylab as pl
import os.path
import StringIO
import base64
from collections import namedtuple


class Report(object):
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.sections = []

    def add_section(self, section):
        self.sections.append(section)

    def to_html(self):
        cur_dir = os.path.dirname(__file__)
        loader = jinja2.FileSystemLoader([os.path.join(cur_dir, 'templates')])
        env = jinja2.Environment(loader=loader)
        template = env.get_template('report.html')
        return template.render(report=self)


class Section(object):
    '''Generic Section'''
    template = 'section.html'

    def __init__(self, title, description, obj):
        self.title = title
        self.description = description
        self.obj = obj


class Table(Section):
    '''Generic Table'''
    template = 'table.html'

    def __init__(self, title, description, headers):
        super(Table, self).__init__(title, description, self)
        self.headers = headers
        self.rows = []

    def add_row(self, row):
        assert len(row) == len(self.headers)
        self.rows.append(row)


Image = namedtuple('Image', 'src')


class Images(Section):
    '''Section of images'''
    template = 'images.html'

    def __init__(self, title, description, colorbar_limits):
        super(Images, self).__init__(title, description, self)
        self._clim = colorbar_limits
        self.images = []

    def add_image(self, data):
        self.images.append(Image(render_image(data, self._clim)))


def _to_base64_img_src(s):
    '''Base64 encoded the binary string and prefix it with
    the proper tag for embedding in html pages
    '''
    srcTag = StringIO.StringIO()
    srcTag.write('data:image/png;base64,')
    srcTag.write(base64.b64encode(s))
    return srcTag.getvalue()


def render_colorbar(colorbar_limits, figsize=(1,3)):
    '''Create a base64 encoded colorbar image'''
    fig = pl.figure(figsize=figsize)
    ax = fig.add_axes([0.05,0.05,0.2,0.9]) # (left, bottom, width, height)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=colorbar_limits[0],
                                vmax=colorbar_limits[1])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    imgData = StringIO.StringIO()
    fig.savefig(imgData, format='png')
    pl.close(fig)
    return _to_base64_img_src(imgData.getvalue())


def render_image(array, colorbar_limits, figsize=(3,3)):
    '''Create a base64 encoded version of data'''
    fig = pl.figure(frameon=False, figsize=figsize)
    ax = pl.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=pl.cm.jet,
              vmin=colorbar_limits[0],
              vmax=colorbar_limits[1])
    img_data = StringIO.StringIO()
    fig.savefig(img_data, format='png')
    pl.close(fig)
    return _to_base64_img_src(img_data.getvalue())
