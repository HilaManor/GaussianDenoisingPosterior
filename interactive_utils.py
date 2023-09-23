import numpy as np
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt


def select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata


def select_impatch(im, manual):
    if manual is not None:
        return {'x1': manual[0], 'x2': manual[1], 'y1': manual[2], 'y2': manual[3]}

    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    if (type(im) == Image.Image and im.mode == 'RGB') or (type(im) != Image.Image and len(im.shape) == 3
                                                          and im.shape[0] == 3):
        plt.imshow(im)
    elif (type(im) == Image.Image and im.mode == 'L') or (type(im) != Image.Image and len(im.shape) > 1):
        plt.imshow(im, cmap='gray')
    else:
        plt.plot(im)
    plt.title("Select a patch, then press <Enter>.")

    rect_select = RectangleSelector(
            ax, select_callback, useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True)  # , drawtype='box')

    def key_checker(event):
        print(event.key)
        # print('aaa')
        if event.key == 'enter':
            selected_coords['x1'], selected_coords['x2'], \
                selected_coords['y1'], selected_coords['y2'] = rect_select.extents
            plt.close()

    selected_coords = {}
    fig.canvas.mpl_connect('key_press_event', key_checker)
    plt.show(block=True)
    # plt.waitforbuttonpress()
    matplotlib.use('Agg')
    return {'x1': int(np.floor(selected_coords['x1'])),
            'x2': int(np.floor(selected_coords['x2'])),
            'y1': int(np.floor(selected_coords['y1'])),
            'y2': int(np.floor(selected_coords['y2']))}
