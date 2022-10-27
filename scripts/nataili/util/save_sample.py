import os

def save_sample(image, filename, sample_path, extension='png', jpg_quality=95, webp_quality=95, webp_lossless=True, png_compression=9):
    path = os.path.join(sample_path, filename + '.' + extension)
    if os.path.exists(path):
        return False
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if extension == 'png':
        image.save(path, format='PNG', compress_level=png_compression)
    elif extension == 'jpg':
        image.save(path, quality=jpg_quality, optimize=True)
    elif extension == 'webp':
        image.save(path, quality=webp_quality, lossless=webp_lossless)
    else:
        return False
    if os.path.exists(path):
        return True
    else:
        return False
