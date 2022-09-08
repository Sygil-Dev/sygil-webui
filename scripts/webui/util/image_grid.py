from util.imports import *
from util.get_font import *

def image_grid(imgs, batch_size, force_n_rows=None, captions=None):
	#print (len(imgs))
	if force_n_rows is not None:
		rows = force_n_rows
	elif defaults.general.n_rows > 0:
		rows = defaults.general.n_rows
	elif defaults.general.n_rows == 0:
		rows = batch_size
	else:
		rows = math.sqrt(len(imgs))
		rows = round(rows)

	cols = math.ceil(len(imgs) / rows)

	w, h = imgs[0].size
	grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

	fnt = get_font(30)

	for i, img in enumerate(imgs):
		grid.paste(img, box=(i % cols * w, i // cols * h))
		if captions and i<len(captions):
			d = ImageDraw.Draw( grid )
			size = d.textbbox( (0,0), captions[i], font=fnt, stroke_width=2, align="center" )
			d.multiline_text((i % cols * w + w/2, i // cols * h + h - size[3]), captions[i], font=fnt, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0), anchor="mm", align="center")

	return grid
