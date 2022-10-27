import math

import PIL


def image_grid(imgs, n_rows=None):
    if n_rows is not None:
        rows = n_rows
    else:
        rows = math.sqrt(len(imgs))
        rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
