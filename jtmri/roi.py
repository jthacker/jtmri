import numpy as np
import time
from kiva.agg import points_in_polygon

from enable.api import Component, ComponentEditor
from traits.api import Array, HasTraits, Instance, Str, List, Property, cached_property, Float, Int, Range
from traitsui.api import Item, HGroup, View, TableEditor, ObjectColumn, HSplit, VSplit, RangeEditor

# Chaco imports
from chaco import default_colormaps
from chaco.api import ArrayPlotData, ArrayDataSource, jet, Plot, LassoOverlay
from chaco.tools.api import LassoSelection, LassoSelection, ZoomTool, PanTool, LineSegmentTool, ImageInspectorTool, ImageInspectorOverlay


SIZE = (800,600)

class Selection(HasTraits):
    name = Str
    mask = Array
    size = Property(depends_on='mask')
    mean = Float
    std  = Float

    def _get_size(self):
        return self.mask.sum()


table_editor = TableEditor(
    columns = [ ObjectColumn(name='name'),
                ObjectColumn(name='size', editable=False),
                ObjectColumn(name='mean', editable=False),
                ObjectColumn(name='std', editable=False)],
    deletable = True,
    auto_size = True,
    orientation = 'vertical',
    edit_view = View( 
                    'name',
                    resizable = True
                )
    )


class MyImageInspectorOverlay(ImageInspectorOverlay):
    def _new_value_updated(self, event):
        if event is None:
            self.text = ""
            if self.visibility == "auto":
                self.visible = False
            return
        elif self.visibility == "auto":
            self.visible = True

        if self.tooltip_mode:
            self.alternate_position = self.image_inspector.last_mouse_position
        else:
            self.alternate_position = None

        d = event
        newstring = ""
        if 'indices' in d:
            newstring += '(%03d, %03d)' % d['indices'] + '\n'
        if 'data_value' in d:
            newstring += "% 0.2g" % d['data_value']

        self.text = newstring
        self.component.request_redraw()



class Model(HasTraits):
    data = Property(depends_on=['rawdata','zindex'])
    rawdata = Array
    plot = Instance(Component)
    selections = List(Selection)
    maxzindex = Property(depends_on='rawdata')
    zindex = Int
    pd = ArrayPlotData()
    
    traits_view = View(
        HSplit(
            VSplit(
                Item('plot', editor=ComponentEditor(), show_label=False),
                Item('zindex', editor=RangeEditor(mode='slider',low=0, is_float=False, high_name='maxzindex'), show_label=False)),
            Item('selections', editor=table_editor, show_label=False, width=50)),
        resizable=True,
        title='Image plot with lasso')

    def _get_data(self):
        return self.rawdata[...,self.zindex]

    def _get_maxzindex(self):
        return self.rawdata.shape[2] - 1

    def _rawdata_changed(self):
        self.pd.set_data("data", self.data)

    def _zindex_changed(self):
        self.pd.set_data('data', self.data)

    def _plot_default(self):
        self.pd.set_data("data", self.data)

        # Create the plot
        plot = Plot(self.pd)
        ybound,xbound = self.data.shape
        img_plot = plot.img_plot("data",
                                 xbounds=(0, xbound),
                                 ybounds=(0, ybound),
                                 colormap=default_colormaps.fix(jet, (-1,1)))[0]

        lasso = MyLasso(component=img_plot, model=self)
        lasso_overlay = LassoOverlay(lasso_selection=lasso, component=img_plot)

        img_plot.tools.append(lasso)
        img_plot.overlays.append(lasso_overlay)
        
        img_inspector = ImageInspectorTool(component=img_plot)
        img_plot.tools.append(img_inspector)
        img_plot.overlays.append(MyImageInspectorOverlay(component=img_plot, 
            image_inspector=img_inspector))
        '''
        pan = PanTool(plot, drag_button="right", constrain_key="shift")
        plot.tools.append(pan)
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)
        plot.overlays.append(LineSegmentTool(plot))
        '''
        return plot


class MyLasso(LassoSelection):
    model = Model
    curr_selection = Instance(Selection)

    def __init__(self, *args, **kwargs):
        super(MyLasso, self).__init__(*args, **kwargs)
        x,y = self.model.data.shape
        self.shape = (x,y)
        self.idxs = np.array([(j,i) for i in range(x) for j in range(y)])
        self.mask = np.zeros(self.model.data.shape, dtype=np.bool)

    def _curr_selection_default(self):
        return None

    def _updated_fired(self):
        if not self.curr_selection:
            self.curr_selection = Selection(name='Selection')
            self.model.selections.append(self.curr_selection)

    def _selection_completed_fired(self):
        inpoly = np.array(points_in_polygon(self.idxs, self._active_selection),dtype=np.bool)
        self.mask = inpoly.reshape(*self.shape)
        masked = np.ma.masked_where(~self.mask, self.model.data)
        self.curr_selection.mean = masked.mean() if masked.any() else 0.0
        self.curr_selection.std = masked.std() if masked.any() else 0.0
        self.curr_selection.mask = self.mask
        self.curr_selection = None


def get_roi(data):
    m = Model(rawdata=data)
    m.configure_traits()
    return m


def test_data(h=200,w=300,d=10):
    x = np.linspace(-np.pi, np.pi, h)[:, np.newaxis, np.newaxis]
    y = np.linspace(-np.pi, np.pi, w)[np.newaxis, :, np.newaxis]
    z = np.linspace(-np.pi, np.pi, d)[np.newaxis, np.newaxis, :]
    return (np.cos(2 * np.pi * 0.1 * x*z) * np.cos(2 * np.pi * 0.9 * z*y))

if __name__=='__main__':
    get_roi(test_data())
