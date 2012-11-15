import numpy as np
import time
from kiva.agg import points_in_polygon

from enable.api import Component, ComponentEditor
from traits.api import Array, HasTraits, Instance, Str, List, Property, cached_property
from traitsui.api import Item, HGroup, View, TableEditor, ObjectColumn

# Chaco imports
from chaco.api import ArrayPlotData, ArrayDataSource, jet, Plot, LassoOverlay
from chaco.tools.api import LassoSelection, LassoSelection


SIZE = (800,600)

class Selection(HasTraits):
    name = Str
    coordinates = Array
    size = Property(depends_on='coordinates')
    mean = Property(depends_on='coordinates')

    def _get_size(self):
        return self.coordinates.size

    @cached_property
    def _get_mean(self):
        return self.coordinates.mean()


table_editor = TableEditor(
    columns = [ ObjectColumn(name='name'),
                ObjectColumn(name='size', editable=False),
                ObjectColumn(name='mean', editable=False)],
    deletable = True,
    auto_size = True,
    orientation = 'vertical',
    edit_view = View( 
                    'name',
                    resizable = True
                )
    )

class Model(HasTraits):
    data = Array
    plot = Instance(Component)
    selections = List(Selection)
    
    traits_view = View(
        HGroup(
            Item('plot', editor=ComponentEditor(), show_label=False),
            Item('selections', editor=table_editor, show_label=False)),
        resizable=True,
        title='Image plot with lasso')

    def _plot_default(self):
        # Create a plot data obect and give it this data
        pd = ArrayPlotData()
        pd.set_data("imagedata", self.data)

        # Create the plot
        plot = Plot(pd)
        ybound,xbound = self.data.shape
        img_plot = plot.img_plot("imagedata",
                                 xbounds=(0, xbound),
                                 ybounds=(0, ybound),
                                 colormap=jet)[0]

        lasso = MyLasso(component=img_plot, model=self)
        lasso_overlay = LassoOverlay(lasso_selection=lasso, component=img_plot)

        img_plot.tools.append(lasso)
        img_plot.overlays.append(lasso_overlay)
        return plot


class MyLasso(LassoSelection):
    model = Model
    curr_selection = Instance(Selection)

    def __init__(self, *args, **kwargs):
        super(MyLasso, self).__init__(*args, **kwargs)
        x,y = self.model.data.shape
        self.idxs = np.array([(i,j) for i in range(x) for j in range(y)])
        self.mask = np.zeros((x,y), dtype=np.bool)

    def _curr_selection_default(self):
        return None

    def _updated_fired(self, new_selection):
        if not self.curr_selection:
            self.curr_selection = Selection(name='Selection')
            self.model.selections.append(self.curr_selection)
       
        inpoly = np.array(points_in_polygon(self.idxs, self._active_selection),dtype=np.bool)
        self.mask = inpoly.reshape(*self.model.data.shape)
        self.curr_selection.coordinates = self.mask

    def _selection_completed_fired(self):
        self.curr_selection = None


def test_data():
    t = np.linspace(-np.pi, np.pi, 200)
    u = np.linspace(-np.pi, np.pi, 300)
    X,Y = np.meshgrid(t,u)
    return np.cos(2 * np.pi * 2 * X) + np.cos(2 * np.pi * 7 * Y)

if __name__=='__main__':
    m = Model(data=test_data())
    m.configure_traits()
