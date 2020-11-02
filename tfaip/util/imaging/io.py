import os
import numpy as np
from PIL import Image

from tfaip.util.imaging.conversion import make_uint8
from tfaip.util.math.hex_byte import hex_byte_converter


def load_image_from_img_file(img_fn: str, img_load_mode='L') -> np.ndarray:
    # open image in s/w mode,  return dim=2
    in_img_pil = load_pil_from_img_file(img_fn, img_load_mode)
    in_img = np.array(in_img_pil, dtype=np.float32)
    return in_img


def load_pil_from_img_file(img_fn: str, img_load_mode='L') -> Image:
    # open image in s/w mode,  return dim=2
    if img_load_mode:
        in_img = Image.open(fp=img_fn).convert(img_load_mode)
    else:
        in_img = Image.open(fp=img_fn)
    return in_img


def save_image_as_hex_text_file(img: np.ndarray, hex_txt_fn: str) -> bool:
    dirname = os.path.dirname(hex_txt_fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif not os.path.isdir(dirname):
        raise Exception(dirname + " exists and isn't a dir")
    hx = hex_byte_converter.bytes2hex2d(make_uint8(img))
    with open(hex_txt_fn, "w") as f:
        f.write(hx)
    return os.path.isfile(hex_txt_fn)


def copy_image_file_into_hex_text_file(img_fn: str, hex_txt_fn: str) -> bool:
    img = load_image_from_img_file(img_fn)
    return save_image_as_hex_text_file(img, hex_txt_fn)


def load_image_from_hex_text_file(hex_txt_fn: str) -> np.ndarray:
    with open(hex_txt_fn, "r") as f:
        hx = f.read()
        return hex_byte_converter.hex2bytes2d(hx)

# def save_image_as_byte_dump(img, byte_dump_fn):
#     # type: (ndarray, unicode) -> bool
#     dirname = os.path.dirname(byte_dump_fn)
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)
#     elif not os.path.isdir(dirname):
#         raise Exception(dirname + " exists and isn't a dir")
#     with open(byte_dump_fn, 'w') as f:
#         f.write(str(' '.join([str(i) for i in img.shape])))
#         f.write('\n')
#         f.write(img.tobytes())
#     return os.path.isfile(byte_dump_fn)
#
#
# def copy_image_file_into_byte_dump(img_fn, byte_dump_fn):
#     # type: (unicode, unicode) -> bool
#     img = load_image_from_img_file(img_fn)
#     return save_image_as_byte_dump(img, byte_dump_fn)
#
#
# def load_image_from_byte_dump(byte_dump_fn):
#     # type: (unicode) -> ndarray
#     with open(byte_dump_fn, 'r') as f:
#         shape_str = f.readline().strip()
#         shape = tuple(np.fromstring(shape_str, dtype=int, sep=' '))
#         size = shape[0] * shape[1]
#         img = Image.frombytes(mode='L', size=shape, data=f.read(size))
#         return np.array(img.getdata()).reshape((img.size[0], img.size[1]))
