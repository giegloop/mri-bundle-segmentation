import numpy as np
from dipy.viz import actor, window, widget

data = np.load('data/diff_data.npy')
data = data[:, :, :, 100]
shape = data.shape

ren = window.Renderer()
image_actor = actor.slicer(data, affine=np.eye(4))

slicer_opacity = .6
image_actor.opacity(slicer_opacity)

ren.add(image_actor)

show_m = window.ShowManager(ren, size=(1200, 900))
show_m.initialize()


def change_slice(obj, event):
    z = int(np.round(obj.get_value()))
    image_actor.display_extent(0, shape[0] - 1,
                               0, shape[1] - 1, z, z)

slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_slice,
                       min_value=0,
                       max_value=shape[2] - 1,
                       value=shape[2] / 2,
                       label="Move slice",
                       right_normalized_pos=(.98, 0.6),
                       size=(120, 0), label_format="%0.lf",
                       color=(1., 1., 1.),
                       selected_color=(0.86, 0.33, 1.))

global size
size = ren.GetSize()


def win_callback(obj, event):
    global size
    if size != obj.GetSize():

        slider.place(ren)
        size = obj.GetSize()

show_m.initialize()

show_m.add_window_callback(win_callback)
show_m.render()
show_m.start()

ren.zoom(1.5)
ren.reset_clipping_range()

del show_m