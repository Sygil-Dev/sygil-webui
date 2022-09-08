from util.imports import *

def get_font(fontsize):
	fonts = ["arial.ttf", "DejaVuSans.ttf"]
	for font_name in fonts:
		try:
			return ImageFont.truetype(font_name, fontsize)
		except OSError:
			pass

	# ImageFont.load_default() is practically unusable as it only supports
	# latin1, so raise an exception instead if no usable font was found
	raise Exception(f"No usable font found (tried {', '.join(fonts)})")
