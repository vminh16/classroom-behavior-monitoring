# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import glob

import cv2
import numpy as np
import math
import paddle
import sys
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from paddle.inference import Config, create_predictor
from python.utils import argsparser, Timer, get_current_memory_mb
from python.benchmark_utils import PaddleInferBenchmark
from python.infer import Detector, print_arguments
try:
    from .attr_infer import AttrDetector
except ImportError:
    from attr_infer import AttrDetector


class SkeletonActionRecognizer(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        threshold (float): The threshold of score for visualization
        window_size(int): Temporal size of skeleton feature.
        random_pad (bool): Whether do random padding when frame length < window_size.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 window_size=100,
                 random_pad=False):
        assert batch_size == 1, "SkeletonActionRecognizer only support batch_size=1 now."
        super(SkeletonActionRecognizer, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold,
            delete_shuffle_pass=True)

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   window_size=cfg['max_frames'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeat number for prediction
        Returns:
            results (dict): 
        '''
        # model prediction
        output_names = self.predictor.get_output_names()
        for i in range(repeats):
            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            np_output = output_tensor.copy_to_cpu()
        result = dict(output=np_output)
        return result

    def predict_skeleton(self, skeleton_list, run_benchmark=False, repeats=1):
        results = []
        for i, skeleton in enumerate(skeleton_list):
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(skeleton)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(skeleton)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(skeleton)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(skeleton)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(skeleton)

            results.append(result)
        return results

    def predict_skeleton_with_mot(self, skeleton_with_mot, run_benchmark=False):
        """
            skeleton_with_mot (dict): includes individual skeleton sequences, which shape is [C, T, K, 1]
                                      and its corresponding track id.
        """

        skeleton_list = skeleton_with_mot["skeleton"]
        mot_id = skeleton_with_mot["mot_id"]
        act_res = self.predict_skeleton(skeleton_list, run_benchmark, repeats=1)
        results = list(zip(mot_id, act_res))
        
        # Debug: Print prediction results for each person (disabled for production)
        # for tracker_id, res in results:
        #     pred_class = res['class'][0]
        #     pred_score = res['score'][0]
        #     print(f"[STGCN Debug] Person {int(tracker_id)}: predicted class={pred_class}, score={pred_score:.4f}")
        
        return results

    def normalize_skeleton(self, skeleton):
        """Normalize skeleton keypoints to match training data format.
        
        Center around shoulder center and normalize by shoulder width.
        This must match the normalization used during STGCN training.
        
        Input: keypoints in relative coordinates (0-1 range within bbox)
        Output: centered and scaled coordinates (similar to training data)
        
        Args:
            skeleton: shape (C=2, T, V, M=1) where C is (x, y)
            
        Returns:
            Normalized skeleton with same shape
        """
        skeleton = skeleton.copy()
        x = skeleton[0]  # (T, V, M) - relative coords 0-1
        y = skeleton[1]
        
        # Debug: print before normalization (disabled)
        # print(f"[Normalize Debug] Before - x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Use shoulder center as reference (indices 5, 6 are left/right shoulders)
        # For 11 keypoint model: 0-nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists
        center_x = (x[:, 5:6, :] + x[:, 6:7, :]) / 2  # (T, 1, M)
        center_y = (y[:, 5:6, :] + y[:, 6:7, :]) / 2
        
        # Center the coordinates
        x = x - center_x
        y = y - center_y
        
        # Calculate shoulder width (in relative coords)
        shoulder_width = np.sqrt(
            (x[:, 5:6, :] - x[:, 6:7, :]) ** 2 +
            (y[:, 5:6, :] - y[:, 6:7, :]) ** 2
        )
        shoulder_width = np.maximum(shoulder_width, 1e-6)  # Avoid division by zero
        shoulder_width = np.mean(shoulder_width)  # Use average width across frames
        
        # Normalize by shoulder width
        x = x / (shoulder_width * 2)
        y = y / (shoulder_width * 2)
        
        skeleton[0] = x
        skeleton[1] = y
        
        return skeleton
    
    def preprocess(self, data):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_lst = []
        
        # Normalize skeleton to match training data format
        # Before: coordinates scaled to coord_size (e.g., 384x512)
        # After: centered around shoulder center, normalized by shoulder width
        data = self.normalize_skeleton(data)
        
        data = action_preprocess(data, preprocess_ops)
        input_lst.append(data)
        input_names = self.predictor.get_input_names()
        inputs = {}
        # inputs['data_batch_0'] = np.stack(input_lst, axis=0).astype('float32')  # Original for PaddleVideo STGCN
        # Support both 'data_batch_0' (PaddleVideo) and 'x' (custom trained model)
        input_data = np.stack(input_lst, axis=0).astype('float32')
        for name in input_names:
            inputs[name] = input_data

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        output_logit = result['output'][0]
        
        # Apply softmax to get probabilities
        exp_logit = np.exp(output_logit - np.max(output_logit))
        probs = exp_logit / np.sum(exp_logit)
        
        # Get predicted class and probability
        pred_class = np.argmax(probs)
        pred_prob = probs[pred_class]
        
        # Threshold for sleeping detection (class 1)
        # Only classify as sleeping if probability > threshold
        sleeping_threshold = 0.8  # Adjust this value (0.5-0.9)
        if pred_class == 1 and pred_prob < sleeping_threshold:
            # Not confident enough, default to normal
            pred_class = 0
            pred_prob = probs[0]
        
        result = {'class': np.array([pred_class]), 'score': np.array([pred_prob])}
        return result


def action_preprocess(input, preprocess_ops):
    """
    input (str | numpy.array): if input is str, it should be a legal file path with numpy array saved.
                               Otherwise it should be numpy.array as direct input.
    return (numpy.array) 
    """
    if isinstance(input, str):
        assert os.path.isfile(input) is not None, "{0} not exists".format(input)
        data = np.load(input)
    else:
        data = input
    for operator in preprocess_ops:
        data = operator(data)
    return data


class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size (int): Temporal size of skeleton feature.
        random_pad (bool): Whether do random padding when frame length < window size. Default: False.
    """

    def __init__(self, window_size=100, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(
                0, self.window_size - T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(
                    T, self.window_size, replace=False).astype('int64')
            else:
                index = np.linspace(0, T, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        return data_pad


def get_test_skeletons(input_file):
    assert input_file is not None, "--action_file can not be None"
    input_data = np.load(input_file)
    if input_data.ndim == 4:
        return [input_data]
    elif input_data.ndim == 5:
        output = list(
            map(lambda x: np.squeeze(x, 0),
                np.split(input_data, input_data.shape[0], 0)))
        return output
    else:
        raise ValueError(
            "Now only support input with shape: (N, C, T, K, M) or (C, T, K, M)")


class DetActionRecognizer(object):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        threshold (float): The threshold of score for action feature object detection.
        display_frames (int): The duration for corresponding detected action.
        skip_frame_num (int): The number of frames for interval prediction. A skipped frame will 
            reuse the result of its last frame. If it is set to 0, no frame will be skipped. Default
            is 0.

    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 display_frames=20,
                 skip_frame_num=0):
        super(DetActionRecognizer, self).__init__()
        self.detector = Detector(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold)
        self.threshold = threshold
        self.frame_life = display_frames
        self.result_history = {}
        self.skip_frame_num = skip_frame_num
        self.skip_frame_cnt = 0
        self.id_in_last_frame = []

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   threshold=cfg['threshold'],
                   display_frames=cfg['display_frames'],
                   skip_frame_num=cfg['skip_frame_num'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def predict(self, images, mot_result):
        if self.skip_frame_cnt == 0 or (not self.check_id_is_same(mot_result)):
            det_result = self.detector.predict_image(images, visual=False)
            result = self.postprocess(det_result, mot_result)
        else:
            result = self.reuse_result(mot_result)

        self.skip_frame_cnt += 1
        if self.skip_frame_cnt >= self.skip_frame_num:
            self.skip_frame_cnt = 0

        return result

    def postprocess(self, det_result, mot_result):
        np_boxes_num = det_result['boxes_num']
        if np_boxes_num[0] <= 0:
            return [[], []]

        mot_bboxes = mot_result.get('boxes')

        cur_box_idx = 0
        mot_id = []
        act_res = []
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]

            # Current now,  class 0 is positive, class 1 is negative.
            action_ret = {'class': 1.0, 'score': -1.0}
            box_num = np_boxes_num[idx]
            boxes = det_result['boxes'][cur_box_idx:cur_box_idx + box_num]
            cur_box_idx += box_num
            isvalid = (boxes[:, 1] > self.threshold) & (boxes[:, 0] == 0)
            valid_boxes = boxes[isvalid, :]

            if valid_boxes.shape[0] >= 1:
                action_ret['class'] = valid_boxes[0, 0]
                action_ret['score'] = valid_boxes[0, 1]
                self.result_history[
                    tracker_id] = [0, self.frame_life, valid_boxes[0, 1]]
            else:
                history_det, life_remain, history_score = self.result_history.get(
                    tracker_id, [1, self.frame_life, -1.0])
                action_ret['class'] = history_det
                action_ret['score'] = -1.0
                life_remain -= 1
                if life_remain <= 0 and tracker_id in self.result_history:
                    del (self.result_history[tracker_id])
                elif tracker_id in self.result_history:
                    self.result_history[tracker_id][1] = life_remain
                else:
                    self.result_history[tracker_id] = [
                        history_det, life_remain, history_score
                    ]

            mot_id.append(tracker_id)
            act_res.append(action_ret)
        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result

    def check_id_is_same(self, mot_result):
        mot_bboxes = mot_result.get('boxes')
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            if tracker_id not in self.id_in_last_frame:
                return False
        return True

    def reuse_result(self, mot_result):
        # This function reusing previous results of the same ID directly.
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            history_cls, life_remain, history_score = self.result_history.get(
                tracker_id, [1, 0, -1.0])

            life_remain -= 1
            if tracker_id in self.result_history:
                self.result_history[tracker_id][1] = life_remain

            action_ret = {'class': history_cls, 'score': history_score}
            mot_id.append(tracker_id)
            act_res.append(action_ret)

        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result


class ClsActionRecognizer(AttrDetector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        threshold (float): The threshold of score for action feature object detection.
        display_frames (int): The duration for corresponding detected action. 
        skip_frame_num (int): The number of frames for interval prediction. A skipped frame will 
            reuse the result of its last frame. If it is set to 0, no frame will be skipped. Default
            is 0.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 display_frames=80,
                 skip_frame_num=0,
                 crop_mode='upper',
                 upper_crop_ratio=0.5):
        super(ClsActionRecognizer, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold)
        self.threshold = threshold
        self.frame_life = display_frames
        self.result_history = {}
        self.skip_frame_num = skip_frame_num
        self.skip_frame_cnt = 0
        self.id_in_last_frame = []
        self.crop_mode = str(crop_mode or 'upper').strip().lower()
        self.upper_crop_ratio = float(upper_crop_ratio)

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   threshold=cfg['threshold'],
                   display_frames=cfg['display_frames'],
                   skip_frame_num=cfg['skip_frame_num'],
                   crop_mode=cfg.get('crop_mode', 'upper'),
                   upper_crop_ratio=cfg.get('upper_crop_ratio', 0.5),
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def predict_with_mot(self, images, mot_result):
        if self.skip_frame_cnt == 0 or (not self.check_id_is_same(mot_result)):
            images = self.crop_person_region(images)
            cls_result = self.predict_image(images, visual=False)["output"]
            result = self.match_action_with_id(cls_result, mot_result)
        else:
            result = self.reuse_result(mot_result)

        self.skip_frame_cnt += 1
        if self.skip_frame_cnt >= self.skip_frame_num:
            self.skip_frame_cnt = 0

        return result

    def crop_person_region(self, images):
        crop_images = []
        for image in images:
            if self.crop_mode in ('full', 'full_body'):
                crop_images.append(image)
                continue
            h = image.shape[0]
            crop_h = int(round(h * self.upper_crop_ratio))
            crop_h = max(1, min(h, crop_h))
            crop_images.append(image[:crop_h, :, :])
        return crop_images

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        im_results = result['output']
        batch_res = []
        for res in im_results:
            action_res = res.tolist()
            for cid, score in enumerate(action_res):
                action_res[cid] = score
            batch_res.append(action_res)
        result = {'output': batch_res}
        return result

    def match_action_with_id(self, cls_result, mot_result):
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            scores_vec = [float(x) for x in cls_result[idx]]
            cls_id_res = int(np.argmax(scores_vec)) if scores_vec else 0
            cls_score_res = float(scores_vec[cls_id_res]) if scores_vec else -1.0

            if cls_score_res < self.threshold:
                history = self.result_history.get(
                    tracker_id, [cls_id_res, self.frame_life, cls_score_res, scores_vec])
                history_cls = history[0] if len(history) > 0 else cls_id_res
                life_remain = history[1] if len(history) > 1 else self.frame_life
                history_score = history[2] if len(history) > 2 else cls_score_res
                history_scores = history[3] if len(history) > 3 else scores_vec

                cls_id_res = history_cls
                cls_score_res = history_score
                scores_vec = history_scores or scores_vec
                life_remain -= 1
                if life_remain <= 0 and tracker_id in self.result_history:
                    del (self.result_history[tracker_id])
                elif tracker_id in self.result_history:
                    self.result_history[tracker_id][1] = life_remain
                else:
                    self.result_history[tracker_id] = [
                        cls_id_res, life_remain, cls_score_res, scores_vec
                    ]
            else:
                self.result_history[tracker_id] = [
                    cls_id_res, self.frame_life, cls_score_res, scores_vec
                ]

            action_ret = {
                'class': cls_id_res,
                'score': cls_score_res,
                'scores': scores_vec,
            }
            mot_id.append(tracker_id)
            act_res.append(action_ret)
        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result

    def check_id_is_same(self, mot_result):
        mot_bboxes = mot_result.get('boxes')
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            if tracker_id not in self.id_in_last_frame:
                return False
        return True

    def reuse_result(self, mot_result):
        # This function reusing previous results of the same ID directly.
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            history = self.result_history.get(tracker_id, [1, 0, -1.0, None])
            history_cls = history[0] if len(history) > 0 else 1
            life_remain = history[1] if len(history) > 1 else 0
            history_score = history[2] if len(history) > 2 else -1.0
            history_scores = history[3] if len(history) > 3 else None

            life_remain -= 1
            if tracker_id in self.result_history:
                self.result_history[tracker_id][1] = life_remain

            action_ret = {
                'class': history_cls,
                'score': history_score,
                'scores': history_scores,
            }
            mot_id.append(tracker_id)
            act_res.append(action_ret)

        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result


def main():
    detector = SkeletonActionRecognizer(
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        threshold=FLAGS.threshold,
        output_dir=FLAGS.output_dir,
        window_size=FLAGS.window_size,
        random_pad=FLAGS.random_pad)
    # predict from numpy array
    input_list = get_test_skeletons(FLAGS.action_file)
    detector.predict_skeleton(input_list, FLAGS.run_benchmark, repeats=10)
    if not FLAGS.run_benchmark:
        detector.det_times.info(average=True)
    else:
        mems = {
            'cpu_rss_mb': detector.cpu_mem / len(input_list),
            'gpu_rss_mb': detector.gpu_mem / len(input_list),
            'gpu_util': detector.gpu_util * 100 / len(input_list)
        }

        perf_info = detector.det_times.report(average=True)
        model_dir = FLAGS.model_dir
        mode = FLAGS.run_mode
        model_info = {
            'model_name': model_dir.strip('/').split('/')[-1],
            'precision': mode.split('_')[-1]
        }
        data_info = {
            'batch_size': FLAGS.batch_size,
            'shape': "dynamic_shape",
            'data_num': perf_info['img_num']
        }
        det_log = PaddleInferBenchmark(detector.config, model_info, data_info,
                                       perf_info, mems)
        det_log('SkeletonAction')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, NPU or XPU"
    assert not FLAGS.use_gpu, "use_gpu has been deprecated, please use --device"

    main()
