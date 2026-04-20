import ast
import yaml
import copy
import argparse
from argparse import ArgumentParser, RawDescriptionHelpFormatter


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            value = yaml.load(v, Loader=yaml.Loader)
            keys = k.split('.')
            cur = config
            for key in keys[:-1]:
                existing = cur.get(key)
                if not isinstance(existing, dict):
                    existing = {}
                    cur[key] = existing
                cur = existing
            cur[keys[-1]] = value
        return config


def argsparser():
    parser = ArgsParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path of configure"),
        required=True)
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Dir of video file, `video_file` has a higher priority.")
    parser.add_argument(
        "--rtsp",
        type=str,
        nargs='+',
        default=None,
        help="list of rtsp inputs, for one or multiple rtsp input.")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--pushurl",
        type=str,
        default="",
        help="url of output visualization stream.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
        "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        "--do_entrance_counting",
        action='store_true',
        help="Whether counting the numbers of identifiers entering "
        "or getting out from the entrance. Note that only support single-class MOT."
    )
    parser.add_argument(
        "--do_break_in_counting",
        action='store_true',
        help="Whether counting the numbers of identifiers break in "
        "the area. Note that only support single-class MOT and "
        "the video should be taken by a static camera.")
    parser.add_argument(
        "--illegal_parking_time",
        type=int,
        default=-1,
        help="Deprecated and unsupported. Vehicle pipeline has been removed; keep this at -1."
    )
    parser.add_argument(
        "--region_type",
        type=str,
        default='horizontal',
        help="Area type for entrance counting or break in counting, 'horizontal' and "
        "'vertical' used when do entrance counting. 'custom' used when do break in counting. "
        "Note that only support single-class MOT, and the video should be taken by a static camera."
    )
    parser.add_argument(
        '--region_polygon',
        nargs='+',
        type=int,
        default=[],
        help="Clockwise point coords (x0,y0,x1,y1...) of polygon of area when "
        "do_break_in_counting. Note that only support single-class MOT and "
        "the video should be taken by a static camera.")
    parser.add_argument(
        "--secs_interval",
        type=int,
        default=2,
        help="The seconds interval to count after tracking")
    parser.add_argument(
        "--draw_center_traj",
        action='store_true',
        help="Whether drawing the trajectory of center")

    return parser


def merge_cfg(args):
    # load config
    with open(args.config) as f:
        pred_config = yaml.safe_load(f)

    def merge(cfg, arg):
        # update cfg from arg directly
        merge_cfg = copy.deepcopy(cfg)
        for k, v in cfg.items():
            if k in arg:
                merge_cfg[k] = arg[k]
            else:
                if isinstance(v, dict):
                    merge_cfg[k] = merge(v, arg)

        return merge_cfg

    def merge_opt(cfg, arg):
        def deep_merge(target, updates, parent_path=""):
            for key, value in updates.items():
                path = "{}.{}".format(parent_path, key) if parent_path else key
                if key not in target:
                    print("No", path, "in config file!")
                    continue
                if isinstance(value, dict):
                    if not isinstance(target[key], dict):
                        print("Config field", path,
                              "is not a dict; replace directly.")
                        target[key] = copy.deepcopy(value)
                        continue
                    deep_merge(target[key], value, path)
                else:
                    target[key] = value

        merge_cfg = copy.deepcopy(cfg)
        # merge opt
        if 'opt' in arg.keys() and arg['opt']:
            deep_merge(merge_cfg, arg['opt'])

        return merge_cfg

    args_dict = vars(args)
    pred_config = merge(pred_config, args_dict)
    pred_config = merge_opt(pred_config, args_dict)

    return pred_config


def print_arguments(cfg):
    print('-----------  Running Arguments -----------')
    buffer = yaml.dump(cfg)
    print(buffer)
    print('------------------------------------------')
