
import argparse, os, sys, glob, re, time
import collections
import yaml
import math
import random
from typing import List, Union, Dict, Callable, Any, Optional, Type, Tuple

import numba

import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageColor

import torch

from frontend.job_manager import JobInfo
from frontend.image_metadata import ImageMetadata

scn2img_cache = {
    "seed": None,
    "cache": {}
}

monocular_depth_estimation = None
def try_loading_monocular_depth_estimation(monocular_depth_estimation_dir = "./src/monocular-depth-estimation/"):
    global monocular_depth_estimation
    if os.path.exists(monocular_depth_estimation_dir):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            except Exception:
                import traceback
                print("Exception during tf.config.experimental.set_virtual_device_configuration:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        try:
            from tensorflow.keras.layers import Layer, InputSpec
            import tensorflow.keras
            # from huggingface_hub import from_pretrained_keras
            # https://stackoverflow.com/a/63631510/798588

            from tensorflow.python.keras.utils import conv_utils

            def normalize_data_format(value):
                if value is None:
                    value = tensorflow.keras.backend.image_data_format()
                data_format = value.lower()
                if data_format not in {'channels_first', 'channels_last'}:
                    raise ValueError('The `data_format` argument must be one of '
                                     '"channels_first", "channels_last". Received: ' +
                                     str(value))
                return data_format


            class BilinearUpSampling2D(Layer):
                def __init__(self, size=(2, 2), data_format=None, **kwargs):
                    super(BilinearUpSampling2D, self).__init__(**kwargs)
                    self.data_format = normalize_data_format(data_format)
                    self.size = conv_utils.normalize_tuple(size, 2, 'size')
                    self.input_spec = InputSpec(ndim=4)

                def compute_output_shape(self, input_shape):
                    if self.data_format == 'channels_first':
                        height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
                        width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
                        return (input_shape[0],
                                input_shape[1],
                                height,
                                width)
                    elif self.data_format == 'channels_last':
                        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
                        width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
                        return (input_shape[0],
                                height,
                                width,
                                input_shape[3])

                def call(self, inputs):
                    input_shape = tensorflow.keras.backend.shape(inputs)
                    if self.data_format == 'channels_first':
                        height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
                        width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
                    elif self.data_format == 'channels_last':
                        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
                        width = self.size[1] * input_shape[2] if input_shape[2] is not None else None

                    return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

                def get_config(self):
                    config = {'size': self.size, 'data_format': self.data_format}
                    base_config = super(BilinearUpSampling2D, self).get_config()
                    return dict(list(base_config.items()) + list(config.items()))

            custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
            monocular_depth_estimation = tf.keras.models.load_model(
                monocular_depth_estimation_dir,
                custom_objects=custom_objects, 
                compile=False
            )
            # todo: load model from pretrained keras into user .cache folder like transformers lib is doing it.
            # 
            # custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
            # custom_objects = {'depth_loss_function': None}
            # monocular_depth_estimation = from_pretrained_keras(
                # "keras-io/monocular-depth-estimation", 
                # custom_objects=custom_objects, compile=False
            # )
            # monocular_depth_estimation = from_pretrained_keras("keras-io/monocular-depth-estimation")
            print('monocular_depth_estimation loaded')
        except Exception:
            import traceback
            print("Error loading monocular_depth_estimation:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)        
    else:
        print(f"monocular_depth_estimation not found at path, please make sure you have cloned \n the repository https://huggingface.co/keras-io/monocular-depth-estimation to {monocular_depth_estimation_dir}")

midas_depth_estimation = None
midas_transforms = None
midas_transform = None
def try_loading_midas_depth_estimation(use_large_model = True):
    global midas_depth_estimation
    global midas_transforms
    global midas_transform
    try:
        if use_large_model:
            midas_depth_estimation = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        else:
            midas_depth_estimation = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        
        device = "cpu"
        midas_depth_estimation.to(device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if use_large_model:
            midas_transform = midas_transforms.default_transform
        else:
            midas_transform = midas_transforms.small_transform
    except Exception:
        import traceback
        print("Error loading midas_depth_estimation:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)        

def try_many(fs, *args, **kwargs):
    for f in fs:
        try:
            return f(*args, **kwargs)
        except:
            pass
    raise Exception("")

def scn2img_define_args():
    parse_arg = {}
    parse_arg["str"]         = lambda x: str(x)
    parse_arg["int"]         = int
    parse_arg["float"]       = float
    parse_arg["bool"]        = lambda s: (s.strip()==str(bool(s)))
    parse_arg["tuple"]       = lambda s: tuple(s.split(",")),
    parse_arg["int_tuple"]   = lambda s: tuple(map(int,s.split(",")))
    parse_arg["float_tuple"] = lambda s: tuple(map(float,s.split(",")))
    parse_arg["degrees"]     = lambda s: float(s) * math.pi / 180
    parse_arg["color"]       = lambda s: try_many([parse_arg["int_tuple"], parse_arg["str"]], s)
    parse_arg["anything"] = lambda s:try_many([
        parse_arg["int_tuple"],
        parse_arg["float_tuple"],
        parse_arg["int"],
        parse_arg["float"],
        parse_arg["tuple"],
        parse_arg["color"],
        parse_arg["str"],
    ],s)
    function_args = {
        "img2img": {
            "prompt"               : "str",
            "image_editor_mode"    : "str",
            "mask_mode"            : "int",
            "mask_blur_strength"   : "float",
            "mask_restore"         : "bool",
            "ddim_steps"           : "int",
            "sampler_name"         : "str",
            "toggles"              : "int_tuple",
            "realesrgan_model_name": "str",
            "n_iter"               : "int",
            "cfg_scale"            : "float",
            "denoising_strength"   : "float",
            "seed"                 : "int",
            "height"               : "int",
            "width"                : "int",
            "resize_mode"          : "int",
            "denoising_strength"   : "float",                
        },
        "txt2img": {
            "prompt"                : "str",
            "ddim_steps"            : "int",
            "sampler_name"          : "str",
            "toggles"               : "int_tuple",
            "realesrgan_model_name" : "str",
            "ddim_eta"              : "float",
            "n_iter"                : "int",
            "batch_size"            : "int",
            "cfg_scale"             : "float",
            "seed"                  : "int",
            "height"                : "int",
            "width"                 : "int",
            "variant_amount"        : "float",
            "variant_seed"          : "int",
        },
        "render_img2img": {
            "select" : "int",
            "variation": "int",
        },
        "render_txt2img": {
            "select" : "int",
            "variation": "int",
        },
        "image": {
            "size"     : "int_tuple",
            "crop"     : "int_tuple",
            "position" : "float_tuple",
            "resize"   : "int_tuple",
            "rotation" : "degrees",
            "color"    : "color",
            "blend"    : "str",
        },
        "render_mask": {
            "mask_value"              : "int",
            "mask_by_color"           : "color",
            "mask_by_color_space"     : "str",
            "mask_by_color_threshold" : "int",
            "mask_by_color_at"        : "int_tuple",
            "mask_is_depth"           : "bool",
            "mask_depth"              : "bool",
            "mask_depth_normalize"    : "bool",
            "mask_depth_model"        : "int",
            "mask_depth_min"          : "float",
            "mask_depth_max"          : "float",
            "mask_depth_invert"       : "bool",
            "mask_open"               : "int",
            "mask_close"              : "int",
            "mask_blur"               : "float",
            "mask_grow"               : "int",
            "mask_shrink"             : "int",
            "mask_invert"             : "bool",
        },
        "render_3d": {
            "transform3d"                      : "bool",
            "transform3d_depth_model"          : "int",
            "transform3d_depth_near"           : "float",
            "transform3d_depth_scale"          : "float",
            "transform3d_from_hfov"            : "degrees",
            "transform3d_from_pose"            : "float_tuple",
            "transform3d_to_hfov"              : "degrees",
            "transform3d_to_pose"              : "float_tuple",
            "transform3d_min_mask"             : "int",
            "transform3d_max_mask"             : "int",
            "transform3d_mask_invert"          : "bool",
            "transform3d_inpaint"              : "bool",
            "transform3d_inpaint_radius"       : "int",
            "transform3d_inpaint_method"       : "int",
            "transform3d_inpaint_restore_mask" : "bool",
        },
        "object": {
            "initial_seed": "int",
        }
    }
    function_args_ext = {
        "image": ["object", "image", "render_mask", "render_3d"],
        "img2img": ["object", "render_img2img", "img2img", "image", "render_mask", "render_3d"],
        "txt2img": ["object", "render_txt2img", "txt2img", "image", "render_mask", "render_3d"],
    }
    return parse_arg, function_args, function_args_ext

def get_scn2img(MemUsageMonitor:Type, save_sample:Callable, get_next_sequence_number:Callable, seed_to_int:Callable, txt2img: Callable, txt2img_defaults: Dict, img2img: Callable, img2img_defaults: Dict, opt: argparse.Namespace = None):
    opt = opt or argparse.Namespace()

    def next_seed(s):
        return random.Random(seed_to_int(s)).randint(0, 2**32 - 1)

    class SeedGenerator:
        def __init__(self, seed):
            self._seed = seed_to_int(seed)
        def next_seed(self):
            seed = self._seed
            self._seed = next_seed(self._seed)
            return seed
        def peek_seed(self):
            return self._seed

    def scn2img(prompt: str, toggles: List[int], seed: Union[int, str, None], fp = None, job_info: JobInfo = None):
        global scn2img_cache
        outpath = opt.outdir_scn2img or opt.outdir or "outputs/scn2img-samples"
        err = False
        seed = seed_to_int(seed)

        prompt = prompt or ''
        clear_cache = 0 in toggles
        output_intermediates = 1 in toggles
        skip_save = 2 not in toggles
        write_info_files = 3 in toggles
        write_sample_info_to_log_file = 4 in toggles
        jpg_sample = 5 in toggles

        os.makedirs(outpath, exist_ok=True)

        if clear_cache or scn2img_cache["seed"] != seed:
            scn2img_cache["seed"] = seed
            scn2img_cache["cache"] = {}

        comments = []
        print_log_lvl = 2
        def gen_log_lines(*args, **kwargs):
            yield (" ".join(map(str, args)))
            for k,v in kwargs.items():
                yield (f"{k} = {v}")
        def log(*args, **kwargs):
            lines = gen_log_lines(*args, **kwargs)
            for line in lines:
                comments.append(line)
        def log_lvl(lvl, *args, **kwargs):
            if (lvl <= print_log_lvl):
                lines = gen_log_lines(*args, **kwargs)
                print("\n".join(lines))
            log(*args, **kwargs)
        def log_trace(*args, **kwargs):
            log_lvl(5,"[TRACE]", *args, **kwargs)
        def log_debug(*args, **kwargs):
            log_lvl(4,"[DEBUG]", *args, **kwargs)
        def log_info(*args, **kwargs):
            log_lvl(3,"[INFO]", *args, **kwargs)
        def log_warn(*args, **kwargs):
            log_lvl(2,"[WARN]", *args, **kwargs)
        def log_err(*args, **kwargs):
            log_lvl(1,"[ERROR]", *args, **kwargs)
        def log_exception(*args, **kwargs):
            log_lvl(0,"[EXCEPTION]", *args, **kwargs)
            import traceback
            log_lvl(0,traceback.format_exc())

        # cache = scn2img_cache["cache"]
        log_info("scn2img_cache")
        log_info(list(scn2img_cache["cache"].keys()))

        def is_seed_invalid(s):
            result = (
                (type(s) != int) 
             or (s == "")
             or (s is None)
            )
            return result
        
        def is_seed_valid(s):
            result =  not is_seed_invalid(s)
            return result

        def vary_seed(s, v):
            s = int(s)
            v = int(v)
            if v == 0: 
                return s
            else:
                return next_seed(s+v)

        if job_info:
            output_images = job_info.images
        else:
            output_images = []

        class SceneObject:
            def __init__(self, func, title, args, depth, children):
                self.func = func
                self.title = title
                self.args = args or collections.OrderedDict()
                self.depth = depth
                self.children = children or []
            def __len__(self):
                return len(self.children)
            def __iter__(self):
                return iter(self.children)
            def __getitem__(self, key):
                if type(key) == int:
                    return self.children[key]
                elif str(key) in self.args:
                    return self.args[str(key)]
                else:
                    return None
            def __setitem__(self, key, value):
                if type(key) == int:
                    self.children[key] = value
                else:
                    self.args[str(key)] = value
            def __contains__(self, key):
                if type(key) == int:
                    return key < len(self.children)
                else:
                    return str(key) in self.args
            def __str__(self):
                return repr(self)
            def __repr__(self):
                args = collections.OrderedDict()
                if len(self.title) > 0:
                    args["title"] = self.title
                args.update(self.args)
                if len(self.children) > 0:
                    args["children"] = self.children
                args = ", ".join(map(lambda kv: f"{str(kv[0])} = {repr(kv[1])}", args.items()))
                return f"{self.func}({args})"
            def cache_hash(self, seed=None, exclude_args=None, exclude_child_args=None, extra=None, child_extra=None):
                exclude_args = exclude_args or set()
                exclude_args = set(exclude_args)
                exclude_child_args = exclude_child_args or set()
                exclude_child_args = set(exclude_child_args)
                if None not in exclude_args:
                    exclude_args.add(None)
                return hash((
                    hash(seed),
                    hash(extra),
                    hash(self.func),
                    hash(tuple([
                        (k,v) for k,v in self.args.items() 
                        if k not in exclude_args
                    ])),
                    hash(tuple([
                        c.cache_hash(
                            seed = seed,
                            exclude_args = exclude_child_args, 
                            exclude_child_args = exclude_child_args,
                            extra = child_extra,
                            child_extra = child_extra
                        ) 
                        for c in self.children
                    ]))
                ))



        parse_arg, function_args, function_args_ext = scn2img_define_args()
        # log_debug("function_args", function_args)

        def parse_scene(prompt, log):

            parse_inline_comment = re.compile(r'(?m)//.+?$') #(?m): $ also matches at before \n
            parse_multiline_comment = re.compile(r'(?s)(^|[^/])/\*.+?\*/') #(?s): . matches \n
            parse_attr = re.compile(r'^\s*([\w_][\d\w_]*)\s*[:=\s]\s*(.+)\s*$')
            parse_heading = re.compile(r'^\s*(#+)([<]?)([>]?)\s*(.*)$') # 

            class Section:
                def __init__(self, depth=0, title="", content=None, children=None):
                    self.depth = depth
                    self.title = title
                    self.lines = []
                    self.content = content or collections.OrderedDict()
                    self.children = children or []
                    self.func = None
                def __repr__(self):
                    return str(self)
                def __str__(self):
                    return "\n".join(
                        [("#"*self.depth) + " " + self.title]
                        + [f"func={self.func}"]
                        + [f"{k}={v}" for k,v in self.content.items()]
                        + list(map(str, self.children))
                    )
            
            def strip_inline_comments(txt):
                while True:
                    txt,replaced = parse_inline_comment.subn("", txt)
                    if replaced == 0:
                        break
                return txt
                
            def strip_multiline_comments(txt):
                while True:
                    txt,replaced = parse_multiline_comment.subn("\1", txt)
                    if replaced == 0:
                        break
                return txt
            
            def strip_comments(txt):
                txt = strip_multiline_comments(txt)
                txt = strip_inline_comments(txt)
                return txt

            def parse_content(lines):
                
                content = collections.OrderedDict()
                for line in lines:
                    # line = strip_inline_comments(line)
                    m = parse_attr.match(line)
                    if m is None:
                        attr = None
                        value = line
                    else:
                        attr = m.group(1)
                        value = m.group(2)
                    
                    is_multi_value = (attr is None)
                    if is_multi_value and attr in content:
                        content[attr].append(value)
                    elif is_multi_value and attr not in content:
                        content[attr] = [value]
                    elif attr not in content:
                        content[attr] = value
                    else:
                        log.append(f"Warn: value for attr {attr} already exists. ignoring {line}.")
                
                return content
                    
            def parse_sections(lines):
                sections = []
                current_section = Section()
                stack = []
                bump_depth = 0
                for line in lines:
                    m = parse_heading.match(line)
                    if m is None:
                        current_section.lines.append(line)
                    else:
                        current_section.content = parse_content(current_section.lines)
                        yield current_section
                        current_section = Section(
                            depth = len(m.group(1)) + bump_depth, 
                            title = m.group(3)
                        )
                        # sections after this will have their depth bumped by number matched '>'.
                        # this allows deep trees while avoiding growing number of '#' by
                        # just using '#> example title' headings
                        bump_depth -= len(m.group(2))
                        bump_depth += len(m.group(3))

                current_section.content = parse_content(current_section.lines)
                yield current_section
            
            def to_trees(sections):
                stack = []
                roots = []
                def insert_section(section):
                    assert(len(stack) == section.depth)
                    if section.depth == 0:
                        roots.append(section)
                    if len(stack) > 0: 
                        parent = stack[len(stack)-1]
                        parent.children.append(section)
                    stack.append(section)

                for section in sections:
                    last_depth = len(stack)-1
                    
                    is_child = section.depth > last_depth
                    is_sibling = section.depth == last_depth
                    is_parental_sibling = section.depth < last_depth
                    if is_child:
                        for d in range(last_depth+1, section.depth, 1):
                            intermediate = Section(depth = d)
                            insert_section(intermediate)
                        
                    elif is_sibling or is_parental_sibling:
                        stack = stack[:section.depth]
                    
                    insert_section(section)
                return roots
            
            def to_scene(trees, depth=0):
                if depth == 0:
                    return SceneObject(
                        func="scn2img",
                        title="",
                        args=None,
                        depth=depth,
                        children=[
                            SceneObject(
                                func="scene",
                                title="",
                                args=None,
                                depth=depth+1,
                                children=[to_scene(tree, depth+2)]
                            )
                            for tree in trees
                        ]
                    )
                else:
                    assert(type(trees) == Section)
                    section = trees
                    has_prompt = "prompt" in section.content
                    has_color = "color" in section.content
                    has_childs = len(section.children) > 0
                    has_input_img = has_childs or has_color
                    func = (
                        "img2img" if (has_input_img and has_prompt) else
                        "txt2img" if (has_prompt) else
                        "image"
                    )            
                    return SceneObject(
                        func=func,
                        title=section.title,
                        args=section.content,
                        depth=depth,
                        children=[
                            to_scene(child, depth+1)
                            for child in section.children
                        ]
                    )
                
            def parse_scene_args(scene):
                image_func_args = function_args["image"]
                scene_func_args = function_args[scene.func] if scene.func in function_args else {}
                extends = function_args_ext[scene.func] if scene.func in function_args_ext else []
                for arg in scene.args.keys():
                    arg_type = "anything"
                    for ext in extends:
                        if arg in function_args[ext]:
                            arg_type = function_args[ext][arg]
                            break
                    try:
                        scene.args[arg] = parse_arg[arg_type](scene.args[arg])
                    except Exception as e:
                        value = scene.args[arg]
                        msg = f"Attribute parsing failed. Expected {arg_type}, got '{value}'."
                        log.append(f"{msg}. Exception: '{str(e)}'")
                for child in scene.children:
                    parse_scene_args(child)
                return scene
            
            prompt = strip_comments(prompt)
            lines = prompt.split("\n")
            sections = parse_sections(lines)
            sections = list(sections)
            trees = to_trees(sections)
            scene = to_scene(trees)
            parse_scene_args(scene)
            
            return scene

        def save_sample_scn2img(img, obj, name, seed):
            if img is None:
                return
            base_count = get_next_sequence_number(outpath)
            filename = "[SEED]_result"
            filename = f"{base_count:05}-" + filename
            filename = filename.replace("[SEED]", str(seed))
            wrapped = SceneObject(
                func=name, 
                title=obj.title, 
                args={"seed":seed}, 
                depth=obj.depth-1, 
                children=[obj]
            )
            info_dict = {
                "prompt": prompt,
                "scene_object": str(wrapped),
                "seed": seed
            }
            metadata = ImageMetadata(prompt=info_dict["scene_object"], seed=seed, width=img.size[0], height=img.size[0])
            ImageMetadata.set_on_image(img, metadata)
            save_sample(img, outpath, filename, jpg_sample, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, False, False)
            if write_info_files:
                filename_i = os.path.join(outpath, filename)
                with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
                    yaml.dump(info_dict, f, allow_unicode=True, width=10000)
            if write_sample_info_to_log_file:
                sample_log_path = os.path.join(outpath, "log.yaml")
                with open(sample_log_path, "a", encoding="utf8") as log_file:
                    yaml.dump(info_dict, log_file, allow_unicode=True, width=10000)
                    log_file.write(" \n")


        def render_scene(output_images, scene, seeds):
            def pose(pos, rotation, center):
                cs, sn = math.cos(rotation), math.sin(rotation)
                return x, y, cs, sn, cy, c

            def pose_mat3(pos=(0,0), rotation=0, center=(0,0)):
                x, y = pos or (0,0)
                cs, sn = math.cos(rotation), math.sin(rotation)
                cx, cy = center or (0,0)
                return (
                    np.array([ # coordinates in parent coordinates
                        [1,0,x],
                        [0,1,y],
                        [0,0,1],
                    ]) @ np.array([ # rotated coordinates with center in origin
                        [cs,-sn,-cx],
                        [+sn,cs,-cy],
                        [0,0,1],
                    ]) # coordinates in pose
                )

            def get_rect(img):
                w, h = img.size
                return np.array([
                    [0, 0], # TL
                    [0, h], # BL
                    [w, h], # BR
                    [w, 0], # TR
                ])

            def transform_points(mat3, pts):
                rot = mat3[:2,:2]
                pos = mat3[:2,2]
                # return rot @ pts.T + pos
                return pts @ rot.T + pos

            def create_image(size, color=None):
                # log_debug("")
                # log_debug("Creating image...", size = type(size), color = color)
                # log_debug("")
                if size is None: return None
                if color is None: color = (0,0,0,0)
                return Image.new("RGBA", size, color) 

            def resize_image(img, size, crop=None):
                if img is None: return None
                if size is None: 
                    return img if (crop is None) else img.crop(box=crop)
                # resize_is_upscaling = (size[0] > img.size[0]) or (size[1] > img.size[1])
                # todo: upscale with realesrgan
                return img.resize(size, box=crop)
            
            def blend_image_at(dst, img, pos, rotation, center, blend_mode):
                if img is None: 
                    return dst
                assert(blend_mode.lower() in ["alpha","mask","add","add_modulo","darker","difference","lighter","logical_and","logical_or","logical_xor","multiply","soft_light","hard_light","overlay","screen","subtract","subtract_modulo"])
                blend_mode = blend_mode.lower()
                # log_debug(f"blend_image_at({dst}, {img}, {pos}, {rotation}, {center})")
                center = center or (img.size[0]*0.5, img.size[1]*0.5)
                pos = pos or ((dst.size[0]*0.5, dst.size[1]*0.5) if dst is not None else None)

                tf = pose_mat3((0,0), rotation)
                rect_points = get_rect(img) - center
                rect_points = transform_points(tf, rect_points)
                min_x = min([p[0] for p in rect_points])
                min_y = min([p[1] for p in rect_points])
                max_x = max([p[0] for p in rect_points])
                max_y = max([p[1] for p in rect_points])
                new_w = max_x - min_x
                new_h = max_y - min_y
                new_size = (int(new_w), int(new_h))

                # default values for pos
                if pos is None and dst is not None:
                    # center img in dst
                    pos = (
                        dst.size[0]*0.5,
                        dst.size[0]*0.5
                    )
                elif pos is None and dst is None:
                    # dst is None, choose pos so that it shows whole img
                    pos = (-min_x, -min_y)
                
                min_x += pos[0]
                min_y += pos[1]
                max_x += pos[0]
                max_y += pos[1]

                if rotation != 0:
                    img = img.rotate(
                        angle = -rotation * (180 / math.pi),
                        expand = True,
                        fillcolor = (0,0,0,0)
                    )

                if (dst is None) and (img.size == new_size):
                    dst = img.copy()
                    # dst = img
                    return dst

                else:
                    if (dst is None):
                        dst = create_image(new_size)
                    dx = int(min_x)
                    dy = int(min_y)
                    sx = -dx if (dx < 0) else 0
                    sy = -dy if (dy < 0) else 0
                    dx = max(0, dx)
                    dy = max(0, dy)
                    # log_debug(f"dest=({dx},{dy}), source=({sx},{sy})")
                    if blend_mode in ["alpha","mask"]:
                        dst.alpha_composite(img, dest=(dx,dy), source=(sx,sy))
                    else:
                        w,h = img.size
                        img_crop = img.crop(box=(sx,sy,w-1,h-1))
                        w,h = img_crop.size
                        dst_crop = dst.crop(box=(dx,dy,dx+w,dy+h))
                        blend_func = getattr(ImageChops, blend_mode)
                        blended = blend_func(dst_crop, img_crop)
                        dst.paste(blended,box=(dx,dy))
                return dst

            def blend_objects(seeds, dst, objects):
                # log_debug("")
                # log_debug(f"blend_objects({dst}, {objects})")
                # log_debug("")
                for obj in reversed(objects):
                    img = render_object(seeds, obj)
                    # if img is None:
                        # log_debug("")
                        # log_debug(f"img is None after render_object in blend_objects({dst}, {objects})")
                        # log_debug("")
                    try:
                        dst = blend_image_at(
                            dst = dst, 
                            img = img, 
                            pos = obj["pos"] or obj["position"] or None, 
                            rotation = obj["rotation"] or obj["rotate"] or obj["angle"] or 0, 
                            center = obj["center"] or None,
                            blend_mode = obj["blend"] if "blend" in obj else "alpha",
                        )
                    except Exception as e:
                        # log_debug("")
                        log_exception(f"Exception! blend_objects({dst}, {objects})")
                        log_err("obj", obj)
                        log_err("img", img)
                        log_err("")
                        raise e

                if dst is not None:
                    dst = dst.copy()
                return dst

            def render_mask(seeds, obj, img, input_mask = None):
                if img is None and input_mask is None: return img

                mask = (
                    img.getchannel("A")
                    if img is not None
                    and input_mask is None 
                    else None
                )
                changed_mask = False

                def combine_masks(old_mask, new_mask, mode):
                    return new_mask

                combine_mode = 1

                if input_mask is not None:
                    mask = input_mask
                    changed_mask = True

                if "mask_value" in obj:
                    new_value = obj["mask_value"]
                    mask.paste( new_value, mask.getbbox() )
                    changed_mask = True

                if ("mask_by_color" in obj or "mask_by_color_at" in obj) and img is not None:
                    img_arr = np.asarray(img.convert("RGB"))
                    color = obj["mask_by_color"]
                    color_at = obj["mask_by_color_at"] or None
                    if color_at is not None:
                        num_points = int(math.floor(len(color_at)/2))
                        points = [
                            (color_at[k*2],color_at[k*2+1]) 
                            for k in range(num_points)
                        ]
                        if len(points) > 0:
                            colors = np.array([img_arr[y,x] for x,y in points])
                            color = tuple(np.round(colors.mean(axis=0)).astype(np.uint8).flatten())
                    colorspace = obj["mask_by_color_space"] or "LAB"
                    threshold = obj["mask_by_color_threshold"] or 15
                    colorspace = colorspace.upper()
                    reference_color = "RGB"
                    if colorspace != "RGB":
                        cvts = {
                            "LAB": cv2.COLOR_RGB2Lab,
                            "LUV": cv2.COLOR_RGB2Luv,
                            "HSV": cv2.COLOR_RGB2HSV,
                            "HLS": cv2.COLOR_RGB2HLS,
                            "YUV": cv2.COLOR_RGB2YUV,
                            "GRAY": cv2.COLOR_RGB2GRAY,
                            "XYZ": cv2.COLOR_RGB2XYZ,
                            "YCrCb": cv2.COLOR_RGB2YCrCb,
                        }
                        rgb = Image.new("RGB", size=(1,1), color=color)
                        rgb_arr = np.asarray(rgb)
                        cvt_arr = cv2.cvtColor(rgb_arr, cvts[colorspace])
                        img_arr = cv2.cvtColor(img_arr, cvts[colorspace])
                        reference_color = cvt_arr[0,0]
                    img_arr = img_arr.astype(np.float32)
                    dist = np.max(np.abs(img_arr - reference_color),axis=2)
                    mask_arr = (dist < threshold).astype(np.uint8) * 255
                    mask = Image.fromarray(mask_arr)
                    changed_mask = True

                if obj["mask_depth"]:
                    mask_depth_min = obj["mask_depth_min"] or 0.2
                    mask_depth_max = obj["mask_depth_max"] or 0.8
                    mask_depth_invert = bool(obj["mask_depth_invert"]) or False
                    mask_is_depth = obj["mask_is_depth"] if "mask_is_depth" in obj else False
                    mask_depth_normalize = obj["mask_depth_normalize"] if "mask_depth_normalize" in obj else True
                    mask_depth_model = int(obj["mask_depth_model"]) if "mask_depth_model" in obj else 1
                    depth = run_depth_estimation(img, mask_depth_model)
                    res = run_depth_filter(depth, mask_depth_min, mask_depth_max, mask_depth_invert, mask_depth_normalize, mask_is_depth)
                    if res is not None:
                        mask = res.resize(img.size)
                        changed_mask = True

                if "mask_open" in obj:
                    mask = mask.filter(ImageFilter.MinFilter(obj["mask_open"]))
                    mask = mask.filter(ImageFilter.MaxFilter(obj["mask_open"]))
                    changed_mask = True

                if "mask_close" in obj:
                    mask = mask.filter(ImageFilter.MaxFilter(obj["mask_close"]))
                    mask = mask.filter(ImageFilter.MinFilter(obj["mask_close"]))
                    changed_mask = True

                if "mask_grow" in obj:
                    mask = mask.filter(ImageFilter.MaxFilter(obj["mask_grow"]))
                    changed_mask = True

                if "mask_shrink" in obj:
                    mask = mask.filter(ImageFilter.MinFilter(obj["mask_shrink"]))
                    changed_mask = True

                if "mask_blur" in obj:
                    mask = mask.filter(ImageFilter.GaussianBlur(obj["mask_blur"]))
                    changed_mask = True

                if obj["mask_invert"]:
                    mask = ImageChops.invert(mask)
                    changed_mask = True

                if changed_mask and img is not None and mask is not None:
                    img.putalpha(mask)
                
                if img is not None:
                    return img
                else:
                    return mask

            # remember output images, to avoid duplicates
            output_image_set = set()

            def output_img(img):
                if img is None: return
                img_id = id(img)
                if img_id in output_image_set:
                    return img
                output_image_set.add(img_id)
                output_images.append(img)

            def render_intermediate(img, obj, name, seed):
                if output_intermediates:
                    output_img(img)
                if not skip_save:
                    save_sample_scn2img(img, obj, name, seed)
                return img

            def render_3d(img, obj):
                if img is None: 
                    return img
                if obj["transform3d"] == True:
                    d2r = math.pi / 180.0
                    depth_model    = obj["transform3d_depth_model"]          if "transform3d_depth_model"          in obj else 1
                    depth_near     = obj["transform3d_depth_near"]           if "transform3d_depth_near"           in obj else 0.1
                    depth_scale    = obj["transform3d_depth_scale"]          if "transform3d_depth_scale"          in obj else 1.0
                    from_hfov      = obj["transform3d_from_hfov"]            if "transform3d_from_hfov"            in obj else (45*d2r)
                    from_pose      = obj["transform3d_from_pose"]            if "transform3d_from_pose"            in obj else (0,0,0, 0,0,0)
                    to_hfov        = obj["transform3d_to_hfov"]              if "transform3d_to_hfov"              in obj else (45*d2r)
                    to_pose        = obj["transform3d_to_pose"]              if "transform3d_to_pose"              in obj else (0,0,0, 0,0,0)
                    min_mask       = obj["transform3d_min_mask"]             if "transform3d_min_mask"             in obj else 128
                    max_mask       = obj["transform3d_max_mask"]             if "transform3d_max_mask"             in obj else 255
                    mask_invert    = obj["transform3d_mask_invert"]          if "transform3d_mask_invert"          in obj else False
                    inpaint        = obj["transform3d_inpaint"]              if "transform3d_inpaint"              in obj else True
                    inpaint_radius = obj["transform3d_inpaint_radius"]       if "transform3d_inpaint_radius"       in obj else 5
                    inpaint_method = obj["transform3d_inpaint_method"]       if "transform3d_inpaint_method"       in obj else 0
                    inpaint_rmask  = obj["transform3d_inpaint_restore_mask"] if "transform3d_inpaint_restore_mask" in obj else False
                    from_pose = list(from_pose)
                    to_pose = list(to_pose)
                    while len(from_pose) < 6: from_pose.append(0)
                    while len(to_pose) < 6: to_pose.append(0)
                    from_pos, from_rpy = from_pose[:3], from_pose[3:6]
                    to_pos, to_rpy = to_pose[:3], to_pose[3:6]
                    hfov0_rad, hfov1_rad = from_hfov, to_hfov
                    tf_world_cam0 = pose3d_rpy(*from_pos, *(deg*d2r for deg in from_rpy))
                    tf_world_cam1 = pose3d_rpy(*to_pos, *(deg*d2r for deg in to_rpy))

                    depth = run_depth_estimation(img, depth_model)
                    img = run_transform_image_3d_simple(img, depth, depth_near, depth_scale, hfov0_rad, tf_world_cam0, hfov1_rad, tf_world_cam1, min_mask, max_mask, mask_invert)
                    if inpaint:
                        mask = img.getchannel("A")
                        img_inpainted = cv2.inpaint(
                            np.asarray(img.convert("RGB")), 
                            255-np.asarray(mask),
                            inpaint_radius,
                            [cv2.INPAINT_TELEA, cv2.INPAINT_NS][inpaint_method]
                        )
                        img = Image.fromarray(img_inpainted).convert("RGBA")
                        if inpaint_rmask:
                            img.putalpha(mask)
                return img

            def render_image(seeds, obj):
                start_seed = seeds.peek_seed()
                img = create_image(obj["size"], obj["color"])
                img = blend_objects(
                    seeds,
                    img,
                    obj.children
                )
                img = render_mask(seeds, obj, img)
                img = resize_image(img, obj["resize"], obj["crop"])
                # if img is None: log_warn(f"result of render_image({obj}) is None")
                img = render_3d(img, obj)
                img = render_intermediate(img, obj, "render_image", start_seed)
                return img

            def prepare_img2img_kwargs(seeds, obj, img):
                # log_trace(f"prepare_img2img_kwargs({obj}, {img})")
                img2img_kwargs = {}
                # img2img_kwargs.update(img2img_defaults)
                func_args = function_args["img2img"]
                for k,v in img2img_defaults.items():
                    if k in func_args:
                        img2img_kwargs[k] = v
                
                if "mask_mode" in img2img_kwargs:
                    img2img_kwargs["mask_mode"] = 1 - img2img_kwargs["mask_mode"]

                if "size" in obj:
                    img2img_kwargs["width"] = obj["size"][0]
                    img2img_kwargs["height"] = obj["size"][1]

                for k,v in func_args.items():
                    if k in obj:
                        img2img_kwargs[k] = obj[k]

                if "toggles" in img2img_kwargs:
                    img2img_kwargs["toggles"] = list(img2img_kwargs["toggles"])

                assert("seed" in img2img_kwargs)
                if "seed" in img2img_kwargs:
                    s = img2img_kwargs["seed"]
                    if is_seed_valid(s):
                        img2img_kwargs["seed"] = int(s)
                    else:
                        img2img_kwargs["seed"] = seeds.next_seed()

                log_info('img2img_kwargs["seed"]', img2img_kwargs["seed"])

                if "variation" in obj:
                    v = obj["variation"]
                    if is_seed_valid(v):
                        s = int(img2img_kwargs["seed"])
                        v = int(v)
                        ns = vary_seed(s, v)
                        log_info(f"Using seed variation {v}: {ns}")
                        img2img_kwargs["seed"] = ns
                    
                img2img_kwargs["job_info"] = job_info
                # img2img_kwargs["job_info"] = None
                img2img_kwargs["fp"] = fp
                img2img_kwargs["init_info"] = img
                if img2img_kwargs["image_editor_mode"] == "Mask":
                    img2img_kwargs["init_info_mask"] = {
                        "image": img.convert("RGB").convert("RGBA"),
                        "mask": img.getchannel("A")
                    }
                    # render_intermediate(img2img_kwargs["init_info_mask"]["mask"].convert("RGBA"), obj, "img2img_init_info_mask", start_seed)
                log_info("img2img_kwargs")
                log_info(img2img_kwargs)

                return img2img_kwargs

            def prepare_txt2img_kwargs(seeds, obj):
                # log_trace(f"prepare_txt2img_kwargs({obj})")
                txt2img_kwargs = {}
                # txt2img_kwargs.update(txt2img_defaults)
                func_args = function_args["txt2img"]
                for k,v in txt2img_defaults.items():
                    if k in func_args:
                        txt2img_kwargs[k] = v

                if "size" in obj:
                    txt2img_kwargs["width"] = obj["size"][0]
                    txt2img_kwargs["height"] = obj["size"][1]

                for k,v in func_args.items():
                    if k in obj:
                        txt2img_kwargs[k] = obj[k]

                if "toggles" in txt2img_kwargs:
                    txt2img_kwargs["toggles"] = list(txt2img_kwargs["toggles"])

                assert("seed" in txt2img_kwargs)
                if "seed" in txt2img_kwargs:
                    s = txt2img_kwargs["seed"]
                    if is_seed_valid(s):
                        txt2img_kwargs["seed"] = int(s)
                    else:
                        txt2img_kwargs["seed"] = seeds.next_seed()

                log_info('txt2img_kwargs["seed"]', txt2img_kwargs["seed"])

                if "variation" in obj:
                    v = obj["variation"]
                    if is_seed_valid(v):
                        s = int(txt2img_kwargs["seed"])
                        v = int(v)
                        ns = vary_seed(s, v)
                        log_info(f"Using seed variation {v}: {ns}")
                        txt2img_kwargs["seed"] = ns

                txt2img_kwargs["job_info"] = job_info
                # txt2img_kwargs["job_info"] = None
                txt2img_kwargs["fp"] = fp

                log_info("txt2img_kwargs")
                log_info(txt2img_kwargs)

                return txt2img_kwargs
                
            def render_img2img(seeds, obj):
                start_seed = seeds.peek_seed()
                global scn2img_cache
                if obj["size"] is None:
                    obj["size"] = (img2img_defaults["width"], img2img_defaults["height"])
                img = create_image(obj["size"], obj["color"])
                img = blend_objects(
                    seeds,
                    img,
                    obj.children
                )
                img = render_mask(seeds, obj, img)
                img = render_intermediate(img, obj, "render_img2img_input", start_seed)

                img2img_kwargs = prepare_img2img_kwargs(seeds, obj, img)

                used_kwargs.append(("img2img", img2img_kwargs))

                # obj_hash = hash(str((img2img_kwargs["seed"],obj)))
                obj_hash = obj.cache_hash(
                    seed = img2img_kwargs["seed"],
                    exclude_args = {"select", "pos", "rotation"}
                )
                if obj_hash not in scn2img_cache["cache"]:
                    if job_info: count_images_before = len(job_info.images)
                    outputs, seed, info, stats = img2img(
                        **img2img_kwargs
                    )
                    if job_info: 
                        # img2img will output into job_info.images.
                        # we want to cache only the new images.
                        # extract new images and remove them from job_info.images.
                        assert(job_info.images == outputs)
                        outputs = job_info.images[count_images_before:]
                        outputs = [img.convert("RGBA") for img in outputs]
                        num_new = len(outputs)
                        # use images.pop so that images list is modified inplace and stays the same object.
                        for k in range(num_new): 
                            job_info.images.pop()
                    scn2img_cache["cache"][obj_hash] = outputs, seed, info, stats
                
                outputs, seed, info, stats = scn2img_cache["cache"][obj_hash]

                for img in outputs:
                    output_img(img)

                log_info("outputs", outputs)

                # select img from outputs
                if len(outputs) > 0:
                    select = obj["select"] or 0
                    img = outputs[select]
                else:
                    # no outputs, so we just use (the input) img without modifying it
                    # img = img
                    pass

                # img = render_mask(seeds, obj, img)
                img = resize_image(img, obj["resize"], obj["crop"])
                if img is None: log_warn(f"result of render_img2img({obj}) is None")
                img = render_3d(img, obj)
                img = render_intermediate(img, obj, "render_img2img", start_seed)
                return img

            def render_txt2img(seeds, obj):
                start_seed = seeds.peek_seed()
                global scn2img_cache

                txt2img_kwargs = prepare_txt2img_kwargs(seeds, obj)

                used_kwargs.append(("txt2img", txt2img_kwargs))

                # obj_hash = hash(str((txt2img_kwargs["seed"],obj)))
                obj_hash = obj.cache_hash(
                    seed = txt2img_kwargs["seed"],
                    exclude_args = {"select", "pos", "rotation"}
                )
                if obj_hash not in scn2img_cache["cache"]:
                    if job_info: count_images_before = len(job_info.images)
                    outputs, seed, info, stats = txt2img(
                        **txt2img_kwargs
                    )
                    if job_info: 
                        # txt2img will output into job_info.images.
                        # we want to cache only the new images.
                        # extract new images and remove them from job_info.images.
                        assert(job_info.images == outputs)
                        outputs = job_info.images[count_images_before:]
                        outputs = [img.convert("RGBA") for img in outputs]
                        num_new = len(outputs)
                        # use images.pop so that images list is modified inplace and stays the same object.
                        for k in range(num_new): 
                            job_info.images.pop()
                    scn2img_cache["cache"][obj_hash] = outputs, seed, info, stats
                
                outputs, seed, info, stats = scn2img_cache["cache"][obj_hash]

                for img in outputs:
                    output_img(img)

                log_info("outputs", outputs)

                # select img from outputs
                if len(outputs) > 0:
                    select = obj["select"] or 0
                    img = outputs[select]
                else:
                    # no outputs, so we use None 
                    img = None

                img = render_mask(seeds, obj, img)
                img = resize_image(img, obj["resize"], obj["crop"])
                if img is None: log_warn(f"result of render_txt2img({obj}) is None")
                img = render_3d(img, obj)
                img = render_intermediate(img, obj, "render_txt2img", start_seed)
                return img

            def render_object(seeds, obj):
                # log_trace(f"render_object({str(obj)})")

                if "initial_seed" in obj:
                    # create new generator rather than resetting current generator,
                    # so that seeds generator from function argument is not changed.
                    seeds = SeedGenerator(obj["initial_seed"])

                if obj.func == "scene":
                    assert(len(obj.children) == 1)
                    return render_object(seeds, obj.children[0])
                elif obj.func == "image":
                    return render_image(seeds, obj)
                elif obj.func == "img2img":
                    return render_img2img(seeds, obj)
                elif obj.func == "txt2img":
                    return render_txt2img(seeds, obj)
                else:
                    msg = f"Got unexpected SceneObject type {obj.func}"
                    comments.append(msg)
                    return None

            def render_scn2img(seeds, obj):
                result = []

                if "initial_seed" in obj:
                    # create new generator rather than resetting current generator,
                    # so that seeds generator from function argument is not changed.
                    seeds = SeedGenerator(obj["initial_seed"])

                if obj.func == "scn2img":
                    # Note on seed generation and for-loops instead of
                    # list-comprehensions:
                    #
                    # For instead of list-comprehension to ensure order as
                    # list-comprehension order is not guaranteed. Seed generator must be
                    # used by children in deterministic order.
                    #
                    # This also applies elsewhere.
                    for child in obj.children: 
                        result.append(render_object(seeds, child))
                else:
                    result.append(render_object(seeds, obj))
                return result

            start_seed = seeds.peek_seed()
            for img in render_scn2img(seeds, scene):
                if output_intermediates:
                    # img already in output, do nothing here
                    pass
                else:
                    output_img(img)

                if skip_save:
                    # individual image save was skipped,
                    # we need to save them now
                    save_sample_scn2img(img, scene, "render_scene", start_seed)

            
            return output_images


        start_time = time.time()

        mem_mon = MemUsageMonitor('MemMon')
        mem_mon.start()

        used_kwargs = []

        scene = parse_scene(prompt, comments)
        log_info("scene")
        log_info(scene)
        # log_info("comments", comments)

        render_scene(output_images, scene, SeedGenerator(seed))
        log_info("output_images", output_images)
        # log_info("comments", comments)

        # comments.append(str(scene))
        mem_max_used, mem_total = mem_mon.read_and_stop()
        time_diff = time.time()-start_time


        output_infos = []
        output_infos.append(("initial_seed", seed))
        excluded_args = set(["job_info", "fp", "init_info", "init_info_mask", "prompt"])
        if len(used_kwargs) > 0:
            for func, kwargs in used_kwargs:
                output_infos.append("\n")
                output_infos.append(("", func))
                output_infos.append(kwargs["prompt"])
                for arg,value in kwargs.items():
                    if arg in excluded_args: continue
                    if value is None: continue
                    if type(value) == dict: continue
                    if type(value) == Image: continue
                    output_infos.append((arg,value))

        full_string = ""
        entities = []
        for output_info in output_infos:
            if type(output_info) == str:
                full_string += output_info
            else:
                assert(type(output_info) is tuple)
                k,v = output_info
                label = f" {k}:" if len(k) > 0 else ""
                entity = {
                    'entity': str(v),
                    'start': len(full_string),
                    'end': len(full_string) + len(label),
                }
                entities.append(entity)
                full_string += label

        info = {
            'text': full_string,
            'entities': entities
        }
        num_prompts = 1
        stats = " ".join([
            f"Took { round(time_diff, 2) }s total ({ round(time_diff/(num_prompts),2) }s per image)",
            f"Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%",
        ])


        return output_images, seed, info, stats, repr(scene)


    return scn2img

def run_monocular_depth_estimation_multi(images, minDepth=10, maxDepth=1000, batch_size=2):
    # https://huggingface.co/keras-io/monocular-depth-estimation
    # https://huggingface.co/spaces/atsantiago/Monocular_Depth_Filter
    global monocular_depth_estimation
    if images is None:
        return None
    if monocular_depth_estimation is None:
        try_loading_monocular_depth_estimation()
    if monocular_depth_estimation is None:
        return None
    if type(images) == Image:
        images = [images]
    loaded_images = []
    for image in images:
        # print("image", image)
        # print("type(image)", type(image))
        #if type(image) is Image:
            # image = np.asarray(image.convert("RGB"))
        try:
            image = image.convert("RGB")
            image = image.resize((640, 480))
        except:
            pass
        image = np.asarray(image)
        x = np.clip(image.reshape(480, 640, 3) / 255, 0, 1)
        loaded_images.append(x)
    loaded_images = np.stack(loaded_images, axis=0)
    images = loaded_images

    # Support multiple RGB(A)s, one RGB(A) image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    if images.shape[3] > 3:   images = images[:,:,:,:3]

    # Compute predictions
    predictions = monocular_depth_estimation.predict(images, batch_size=batch_size)

    def depth_norm(x, maxDepth):
        return maxDepth / x

    # Put in expected range
    # print("Max Depth:", np.amax(predictions), maxDepth)
    # print("Min Depth:", np.amin(predictions), minDepth)
    depths = np.clip(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
    return depths

def run_monocular_depth_estimation_single(image, minDepth=10, maxDepth=1000):
    depth = run_monocular_depth_estimation_multi([image], minDepth, maxDepth)[0][:,:,0]
    return depth

def run_Monocular_Depth_Filter_multi(images, filter_min_depth:float, filter_max_depth:float, invert:bool, normalize_depth:bool, mask_is_depth:bool, **kwargs):
    # https://huggingface.co/spaces/atsantiago/Monocular_Depth_Filter
    depths = run_monocular_depth_estimation_multi(images, **kwargs)
    if depths is None: 
        return None
    n,h,w,c = depths.shape
    # print("run_Monocular_Depth_Filter n,h,w,c", n,h,w,c)
    outputs = []
    for k in range(n):
        depth = depths[k][:,:,0]
        mask = run_depth_filter(depth, filter_min_depth, filter_max_depth, invert, normalize_depth, mask_is_depth)
        outputs.append(mask)
    return outputs

def run_Monocular_Depth_Filter_single(image, filter_min_depth:float, filter_max_depth:float, invert:bool, normalize_depth:bool, mask_is_depth:bool, **kwargs):
    depths = run_Monocular_Depth_Filter_multi([image], filter_min_depth, filter_max_depth, invert, normalize_depth, mask_is_depth, **kwargs)
    return depths[0]


def run_midas_depth_estimation(image):
    global midas_depth_estimation
    global midas_transform
    if image is None:
        return None
    if midas_depth_estimation is None or midas_transform is None:
        try_loading_midas_depth_estimation()
    if midas_depth_estimation is None or midas_transform is None:
        return None

    image = image.convert("RGB")
    image = np.asarray(image) 

    device = "cpu"
    input_batch = midas_transform(image).to(device)
    with torch.no_grad():
        prediction = midas_depth_estimation(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    depth = 1 - output / np.max(output)
    return depth

def run_midas_depth_filter(image, filter_min_depth:float, filter_max_depth:float, invert:bool, normalize_depth:bool, mask_is_depth:bool):
    depth = run_midas_depth_estimation(image)

    return run_depth_filter(depth, filter_min_depth, filter_max_depth, invert, normalize_depth, mask_is_depth)


def run_depth_filter(depth: np.ndarray, filter_min_depth:float, filter_max_depth:float, invert:bool, normalize_depth:bool, mask_is_depth:bool):
    if depth is None:
        return None

    if normalize_depth:
        depth = depth - np.min(depth)
        depth = depth / np.max(depth)

    if mask_is_depth:
        depth = (depth - filter_min_depth) * (1.0/(filter_max_depth - filter_min_depth))
        depth[depth < 0] = 0
        depth[depth > 1] = 1
        mask = (depth*255).astype(np.uint8)
    else:
        filt_arr_min = (depth > filter_min_depth)
        filt_arr_max = (depth < filter_max_depth)
        mask = np.logical_and(filt_arr_min, filt_arr_max).astype(np.uint8) * 255

    if invert:
        mask = 255-mask

    mask = Image.fromarray(mask,"L")

    return mask

def run_depth_estimation(image:Image, model_idx:int):
    funcs_depth_estimation = [run_monocular_depth_estimation_single, run_midas_depth_estimation]
    func_depth_estimation = funcs_depth_estimation[model_idx]
    depth = func_depth_estimation(image)
    return depth

@numba.jit
def depth_reprojection(xyz:np.ndarray, depth:np.ndarray, depth_scale:float, fx:float, fy:float, cx:float, cy:float):
    h,w = depth.shape[:2]
    for v in range(h):
        y = fy*(v - cy)
        for u in range(w):
            x = fx*(u - cx)
            z = depth[v,u] * depth_scale
            xyz[v,u,0] = x*z
            xyz[v,u,1] = y*z
            xyz[v,u,2] = z

def run_3d_estimation(depth:np.ndarray, depth_scale:float=1, hfov_rad:float=60*math.pi/180):
    pass
    h,w = depth.shape[:2]
    cam_info = CameraInfo((h,w), hfov_rad)
    xyz = np.empty(shape=(h, w, 3), dtype=np.float32)
    depth_reprojection(xyz, depth, depth_scale, cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy)
    return xyz

@numba.jit
def transform_image_3d(img_out:np.ndarray, img_in:np.ndarray, depth:np.ndarray, depth_near:float, depth_scale:float,
        fx0:float, fy0:float, cx0:float, cy0:float,
        fx1:float, fy1:float, cx1:float, cy1:float, 
        rot_cam1_cam0: np.ndarray, offset_cam1_cam0: np.ndarray,
        min_mask:int, max_mask:int):
    # assert(img_in.shape[2] == 4)
    # assert(img_out.shape[2] == 4)
    # assert(len(depth.shape) == 2)
    # (u0,v0)  : 2d pixel position in img_in
    # pos_cam0 : 3d pixel position in cam0 coordinate system
    # pos_cam1 : 3d pixel position in cam1 coordinate system
    # (u1,v1)  : 2d pixel position in img_out
    m00 = rot_cam1_cam0[0,0]
    m01 = rot_cam1_cam0[0,1]
    m02 = rot_cam1_cam0[0,2]
    m10 = rot_cam1_cam0[1,0]
    m11 = rot_cam1_cam0[1,1]
    m12 = rot_cam1_cam0[1,2]
    m20 = rot_cam1_cam0[2,0]
    m21 = rot_cam1_cam0[2,1]
    m22 = rot_cam1_cam0[2,2]
    h0 = int(depth.shape[0])
    w0 = int(depth.shape[1])
    h1 = int(img_out.shape[0])
    w1 = int(img_out.shape[1])
    for v0 in range(h0):
        y0_ = fy0*(v0 - cy0)
        for u0 in range(w0):
            r,g,b,a = img_in[v0,u0]
            # img_out[v0,u0,0] = r
            # img_out[v0,u0,1] = g
            # img_out[v0,u0,2] = b
            # img_out[v0,u0,3] = a
            # continue
            # if not (min_mask <= a <= max_mask): continue
            x0_ = fx0*(u0 - cx0)
            z0 = depth_near + depth[v0,u0] * depth_scale
            x0 = x0_ * z0
            y0 = y0_ * z0
            x1 = offset_cam1_cam0[0] + m00*x0 + m01*y0 + m02*z0
            y1 = offset_cam1_cam0[1] + m10*x0 + m11*y0 + m12*z0
            z1 = offset_cam1_cam0[2] + m20*x0 + m21*y0 + m22*z0
            # pos_cam0 = (x0*z0,y0*z0,z0)
            # pos_cam1 = offset_cam1_cam0 + rot_cam1_cam0 @ pos_cam0
            # x1,y1,z1 = pos_cam1
            if z1 <= 0: continue 
            u1 = int(0.5 + (x1/(z1*fx1))+cx1)
            v1 = int(0.5 + (y1/(z1*fy1))+cy1)
            if u1 < 0: u1 = 0
            if u1 >= w1: u1 = w1-1
            if v1 < 0: v1 = 0
            if v1 >= h1: v1 = h1-1
            # if not (0 <= u1 < w1): continue
            # if not (0 <= v1 < h1): continue
            img_out[v1,u1,0] = r
            img_out[v1,u1,1] = g
            img_out[v1,u1,2] = b
            img_out[v1,u1,3] = a

class CameraInfo:
    def __init__(self, image_size:Tuple[int,int], hfov_rad:float=60*math.pi/180, pose:np.ndarray=None):
        self.width = image_size[0]
        self.height = image_size[1]
        self.aspect_ratio = self.width * (1.0 / self.height)
        self.hfov_rad = hfov_rad
        self.vfov_rad = self.hfov_rad / self.aspect_ratio
        half_width = self.width * 0.5
        half_height = self.width * 0.5
        self.fx = math.tan(self.hfov_rad*0.5) / half_width
        self.fy = math.tan(self.vfov_rad*0.5) / half_height
        self.cx = half_width
        self.cy = half_height
        self.pose = pose if pose is not None else np.eye(4)
        assert(self.pose.shape==(4,4))
        
def run_transform_image_3d(image:Image, depth:np.ndarray, depth_near:float, depth_scale:float, from_caminfo: CameraInfo, to_caminfo: CameraInfo, min_mask:int, max_mask:int, mask_invert:bool):
    if image is None: return None
    h,w = image.size
    image_in = np.asarray(image.convert("RGBA"))
    image_out = np.zeros(shape=(h,w,4),dtype=np.uint8)
    tf_world_cam0 = from_caminfo.pose
    tf_world_cam1 = to_caminfo.pose
    tf_cam1_world = affine_inv(tf_world_cam1)
    tf_cam1_cam0 = tf_cam1_world @ tf_world_cam0
    rot_cam1_cam0 = tf_cam1_cam0[:3,:3]
    offset_cam1_cam0 = tf_cam1_cam0[:3,3]
    # print("depth_scale", depth_scale)
    # print("from_caminfo.fx", from_caminfo.fx)
    # print("from_caminfo.fy", from_caminfo.fy)
    # print("from_caminfo.cx", from_caminfo.cx)
    # print("from_caminfo.cy", from_caminfo.cy)
    # print("to_caminfo.fx", to_caminfo.fx)
    # print("to_caminfo.fy", to_caminfo.fy)
    # print("to_caminfo.cx", to_caminfo.cx)
    # print("to_caminfo.cy", to_caminfo.cy)
    # print("rot_cam1_cam0", rot_cam1_cam0)
    # print("offset_cam1_cam0", offset_cam1_cam0)
    # print("min_mask", min_mask)
    # print("max_mask", max_mask)
    
    transform_image_3d(
        image_out, image_in, depth, depth_near, depth_scale, 
        from_caminfo.fx, from_caminfo.fy, from_caminfo.cx, from_caminfo.cy, 
        to_caminfo.fx, to_caminfo.fy, to_caminfo.cx, to_caminfo.cy, 
        rot_cam1_cam0, offset_cam1_cam0,
        min_mask, max_mask
    )
    if mask_invert:
        image_out[:,:,3] = 255 - image_out[:,:,3]
    return Image.fromarray(image_out,"RGBA")

def run_transform_image_3d_simple(image:Image, depth:np.ndarray, depth_near:float, depth_scale:float, 
        hfov0_rad:float, tf_world_cam0: np.ndarray,
        hfov1_rad:float, tf_world_cam1: np.ndarray,
        min_mask:int, max_mask:int, mask_invert:bool):
    from_caminfo = CameraInfo(image.size, hfov0_rad, tf_world_cam0)
    to_caminfo = CameraInfo(image.size, hfov1_rad, tf_world_cam1)
    return run_transform_image_3d(image, depth, depth_near, depth_scale, from_caminfo, to_caminfo, min_mask, max_mask, mask_invert)

def translation3d(x,y,z):
    return np.array([
        [1,0,0,x],
        [0,1,0,y],
        [0,0,1,z],
        [0,0,0,1],
    ])

def rotation3d_x(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [1,0,0,0],
        [0,cs,-sn,0],
        [0,+sn,cs,0],
        [0,0,0,1],
    ])
def rotation3d_y(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [cs,0,+sn,0],
        [0,1,0,0],
        [-sn,0,cs,0],
        [0,0,0,1],
    ])
def rotation3d_z(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [cs,-sn,0,0],
        [+sn,cs,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])

def rotation3d_rpy(roll, pitch, yaw):
    # Diebel, J. (2006). Representing attitude: Euler angles, unit quaternions, and rotation vectors. Matrix, 58(15-16), 1-35.
    # (the paper uses inverse transformations to ours, i.e. transformations from world to body)
    # euler-1-2-3 scheme 
    
    # transforms from body to world
    return rotation3d_z(yaw) @ rotation3d_y(pitch) @ rotation3d_x(roll)

def rpy_from_rotation3d(mat):
    # Diebel, J. (2006). Representing attitude: Euler angles, unit quaternions, and rotation vectors. Matrix, 58(15-16), 1-35.
    # (the paper uses inverse transformations to ours, i.e. transformations from world to body)
    # euler-1-2-3 scheme 
    matT = mat.T
    roll = np.arctan2(matT[1,2], matT[2,2])
    pitch = -np.arcsin(matT[0,2])
    yaw = np.arctan2(matT[0,1], matT[0,0])

    return np.array([roll,pitch,yaw])

def affine_inv(mat44):
    rot=mat44[:3,:3]
    trans=mat44[:3,3]
    inv_rot=rot.T
    inv_trans=-inv_rot@trans
    return pose3d(inv_rot, inv_trans)

def pose3d(rotation, translation):
    mat44 = np.zeros(shape=(4,4),dtype=rotation.dtype)
    mat44[:3,:3] = rotation
    mat44[:3,3] = translation
    return mat44

def pose3d_rpy(x, y, z, roll, pitch, yaw):
    """returns transformation matrix which transforms from pose to world"""
    return translation3d(x,y,z) @ rotation3d_rpy(roll, pitch, yaw)
