from traits.api import Any, Bool, CFloat, CInt, HasTraits, Float, Array, Int, Trait, Callable
from chaco.default_colormaps import jet
from chaco.api import ArrayPlotData, Plot, GridPlotContainer, \
                                 BaseTool, DataRange1D
from chaco.tools.api import LineInspector, ZoomTool

from enable.api import Window
from enable.example_support import DemoFrame, demo_main

import numpy as np

class Model(HasTraits):
    npts_x = CInt(0)
    npts_y = CInt(0)
    npts_z = CInt(0)

    minval = Float
    maxval = Float

    xs = Array
    ys = Array
    zs = Array

    vals = Array

    def set_data(self,data):
        '''data should be a 3D numpy array'''
        assert data.ndim == 3
        
        self.npts_x,self.npts_y,self.npts_z = data.shape
        self.vals = data
        self.xs = np.arange(self.npts_x)
        self.ys = np.arange(self.npts_y)
        self.zs = np.arange(self.npts_z)

        # Generate a cube of values by using newaxis to span new dimensions
        self.minval = np.nanmin(self.vals)
        self.maxval = np.nanmax(self.vals)


class ImageIndexTool(BaseTool):
    """ A tool to set the slice of a cube based on the user's mouse movements
    or clicks.
    """

    # This callback will be called with the index into self.component's
    # index and value:
    #     callback(tool, x_index, y_index)
    # where *tool* is a reference to this tool instance.  The callback
    # can then use tool.token.
    callback = Any()

    # This callback (if it exists) will be called with the integer number
    # of mousewheel clicks
    wheel_cb = Any()

    # This token can be used by the callback to decide how to process
    # the event.
    token  = Any()

    # Whether or not to update the slice info; we enter select mode when
    # the left mouse button is pressed and exit it when the mouse button
    # is released
    # FIXME: This is not used right now.
    select_mode = Bool(False)

    def normal_left_down(self, event):
        self._update_slices(event)

    def normal_right_down(self, event):
        self._update_slices(event)

    def normal_mouse_move(self, event):
        if event.left_down or event.right_down:
            self._update_slices(event)

    def _update_slices(self, event):
            plot = self.component
            ndx = plot.map_index((event.x, event.y), index_only=True)
            if ndx:
                self.callback(self, *ndx)

    def normal_mouse_wheel(self, event):
        if self.wheel_cb is not None:
            self.wheel_cb(self, event.mouse_wheel)


class PlotFrame(DemoFrame):
    # These are the indices into the cube that each of the image plot views
    # will show; the default values are non-zero just to make it a little
    # interesting.
    slice_x = 5
    slice_y = 5
    slice_z = 5

    num_levels = Int(15)
    colormap = Any
    colorcube = Any

    #---------------------------------------------------------------------------
    # Private Traits
    #---------------------------------------------------------------------------

    _cmap = Trait(jet, Callable)

    def _index_callback(self, tool, x_index, y_index):
        plane = tool.token
        if plane == "xy":
            self.slice_x = x_index
            self.slice_y = y_index
        elif plane == "yz":
            # transposed because the plot is oriented vertically
            self.slice_z = x_index
            self.slice_y = y_index
        elif plane == "xz":
            self.slice_x = x_index
            self.slice_z = y_index
        else:
            warnings.warn("Unrecognized plane for _index_callback: %s" % plane)
        self._update_images()
        self.center.invalidate_and_redraw()
        self.right.invalidate_and_redraw()
        self.bottom.invalidate_and_redraw()
        return

    def _wheel_callback(self, tool, wheelamt):
        plane_slice_dict = {"xy": ("slice_z", 2),
                            "yz": ("slice_x", 0),
                            "xz": ("slice_y", 1)}
        attr, shape_ndx = plane_slice_dict[tool.token]
        val = getattr(self, attr)
        max = self.model.vals.shape[shape_ndx]
        if val + wheelamt > max:
            setattr(self, attr, max-1)
        elif val + wheelamt < 0:
            setattr(self, attr, 0)
        else:
            setattr(self, attr, val + wheelamt)

        self._update_images()
        self.center.invalidate_and_redraw()
        self.right.invalidate_and_redraw()
        self.bottom.invalidate_and_redraw()
        return

    def __init__(self, data, *args, **kwargs):
        self.model = Model()
        self.model.setdata = data

    def _create_window(self):
        # Create the model
        cmap = jet

        self._update_model(cmap)

        # Create the plot
        self.plotdata = ArrayPlotData()
        self._update_images()

        # Center Plot
        centerplot = Plot(self.plotdata, padding=0)
        imgplot = centerplot.img_plot("xy",
                                xbounds=(model.xs[0], model.xs[-1]),
                                ybounds=(model.ys[0], model.ys[-1]),
                                colormap=cmap)[0]
        self._add_plot_tools(imgplot, "xy")
        self.center = imgplot

        # Right Plot
        rightplot = Plot(self.plotdata, width=300, resizable="v", padding=0)
        rightplot.value_range = centerplot.value_range
        imgplot = rightplot.img_plot("yz",
                                xbounds=(model.zs[0], model.zs[-1]),
                                ybounds=(model.ys[0], model.ys[-1]),
                                colormap=cmap)[0]
        self._add_plot_tools(imgplot, "yz")
        self.right = imgplot

        # Bottom Plot
        bottomplot = Plot(self.plotdata, height=350, resizable="h", padding=0)
        bottomplot.index_range = centerplot.index_range
        imgplot = bottomplot.img_plot("xz",
                                xbounds=(model.xs[0], model.xs[-1]),
                                ybounds=(model.zs[0], model.zs[-1]),
                                colormap=cmap)[0]
        self._add_plot_tools(imgplot, "xz")
        self.bottom = imgplot

        # Create Container and add all Plots
        container = GridPlotContainer(padding=10, fill_padding=True,
                                      bgcolor="white", use_backbuffer=True,
                                      shape=(2,2), spacing=(12,12))
        container.add(centerplot)
        container.add(rightplot)
        container.add(bottomplot)

        self.container = container
        return Window(self, -1, component=container)

    def _add_plot_tools(self, imgplot, token):
        """ Add LineInspectors, ImageIndexTool, and ZoomTool to the image plots. """

        imgplot.overlays.append(ZoomTool(component=imgplot, tool_mode="box",
                                           enable_wheel=False, always_on=False))
        imgplot.overlays.append(LineInspector(imgplot, axis="index_y", color="white",
            inspect_mode="indexed", write_metadata=True, is_listener=True))
        imgplot.overlays.append(LineInspector(imgplot, axis="index_x", color="white",
            inspect_mode="indexed", write_metadata=True, is_listener=True))
        imgplot.tools.append(ImageIndexTool(imgplot, token=token,
            callback=self._index_callback, wheel_cb=self._wheel_callback))

    def _update_model(self, cmap):
        range = DataRange1D(low=np.amin(self.model.vals),
                            high=np.amax(self.model.vals))
        self.colormap = cmap(range)
        self.colorcube = (self.colormap.map_screen(self.model.vals) * 255).astype(np.uint8)

    def _update_images(self):
        """ Updates the image data in self.plotdata to correspond to the
        slices given.
        """
        cube = self.colorcube
        pd = self.plotdata
        # These are transposed because img_plot() expects its data to be in
        # row-major order
        pd.set_data("xy", np.transpose(cube[:, :, self.slice_z], (1,0,2)))
        pd.set_data("xz", np.transpose(cube[:, self.slice_y, :], (1,0,2)))
        pd.set_data("yz", cube[self.slice_x, :, :])


def plot(data):
    #demo_main(PlotFrame, size=(800,700), title="Cube analyzer")
    frame = PlotFrame(data, size=(800,700))
    frame.configure_traits()


