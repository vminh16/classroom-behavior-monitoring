"""
Sleeping Detection Module V2 - Cải tiến cho môi trường lớp học
Phát hiện ngủ gục dựa trên phân tích keypoint pose từ TinyPose

Cải tiến so với V1:
- Sử dụng shoulder_width thay vì body_height (phù hợp khi hông bị che)
- Thêm confidence filter cho keypoints
- Thêm head_tilt feature
- Tối ưu thresholds cho môi trường lớp học

Author: GitHub Copilot
"""

import numpy as np
from collections import defaultdict, deque
import time


class SleepingActionRecognizerV2:
    """
    Nhận diện hành động ngủ gục dựa trên keypoint analysis - V2
    
    Tối ưu cho môi trường lớp học:
    - Chỉ sử dụng keypoints đầu + vai (6 điểm chính)
    - Không phụ thuộc vào hông/chân (thường bị che bởi bàn)
    
    Keypoint indices (COCO format - sử dụng 7 điểm đầu):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    
    # Key indices cần kiểm tra confidence
    KEY_POINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
    
    def __init__(self,
                 head_down_threshold=0.10,      # Ngưỡng đầu cúi (dùng shoulder_width)
                 head_tilt_threshold=0.35,      # Ngưỡng đầu nghiêng
                 duration_threshold=2.5,        # Giây để xác nhận ngủ  
                 motion_threshold=0.015,        # Ngưỡng motion
                 min_confidence=0.4,            # Confidence tối thiểu cho keypoints
                 buffer_size=40,                # Số frame buffer
                 display_frames=120,            # Số frame hiển thị kết quả
                 smoothing_window=5):           # Cửa sổ làm mượt
        """
        Args:
            head_down_threshold: Ngưỡng để xác định đầu cúi (normalize theo shoulder_width)
            head_tilt_threshold: Ngưỡng để xác định đầu nghiêng sang bên
            duration_threshold: Thời gian tối thiểu (giây) để xác định đang ngủ
            motion_threshold: Ngưỡng chuyển động để xác định "tĩnh"
            min_confidence: Confidence tối thiểu cho keypoint để được sử dụng
            buffer_size: Số frame lưu trữ để phân tích temporal
            display_frames: Số frame duy trì hiển thị kết quả
            smoothing_window: Số frame để làm mượt predictions
        """
        self.head_down_threshold = head_down_threshold
        self.head_tilt_threshold = head_tilt_threshold
        self.duration_threshold = duration_threshold
        self.motion_threshold = motion_threshold
        self.min_confidence = min_confidence
        self.buffer_size = buffer_size
        self.display_frames = display_frames
        self.smoothing_window = smoothing_window
        
        # State tracking cho mỗi person ID
        self.keypoint_history = defaultdict(lambda: deque(maxlen=buffer_size))
        self.feature_history = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.head_down_start = {}  # {person_id: start_time}
        self.sleeping_status = {}  # {person_id: True/False}
        self.result_cache = {}     # {person_id: (class, score, life_remain)}
        
        # FPS estimation
        self.fps = 25.0  # Default
        
    @classmethod
    def init_with_cfg(cls, args, cfg):
        """Initialize từ config file"""
        return cls(
            head_down_threshold=cfg.get('head_down_threshold', 0.10),
            head_tilt_threshold=cfg.get('head_tilt_threshold', 0.35),
            duration_threshold=cfg.get('duration_threshold', 2.5),
            motion_threshold=cfg.get('motion_threshold', 0.015),
            min_confidence=cfg.get('min_confidence', 0.4),
            buffer_size=cfg.get('buffer_size', 40),
            display_frames=cfg.get('display_frames', 120),
            smoothing_window=cfg.get('smoothing_window', 5)
        )
    
    def set_fps(self, fps):
        """Cập nhật FPS từ video"""
        self.fps = max(fps, 1.0)
    
    def _check_keypoint_confidence(self, keypoints):
        """
        Kiểm tra confidence của các keypoints quan trọng
        
        Returns:
            bool: True nếu tất cả keypoints quan trọng đều đủ confidence
        """
        for idx in self.KEY_POINTS:
            if len(keypoints[idx]) >= 3:
                if keypoints[idx][2] < self.min_confidence:
                    return False
            else:
                # Không có confidence score, skip check
                pass
        return True
    
    def _extract_key_points(self, keypoints):
        """
        Trích xuất các điểm quan trọng với confidence filtering
        
        Args:
            keypoints: numpy array shape (17, 3) hoặc (17, 2) - [x, y, score]
        
        Returns:
            dict với các điểm đã xử lý, hoặc None nếu không đủ điều kiện
        """
        if keypoints is None or len(keypoints) < 7:
            return None
        
        # Kiểm tra confidence
        if not self._check_keypoint_confidence(keypoints):
            return None
        
        # Lấy tọa độ các điểm quan trọng
        nose = np.array(keypoints[self.NOSE][:2], dtype=np.float32)
        left_eye = np.array(keypoints[self.LEFT_EYE][:2], dtype=np.float32)
        right_eye = np.array(keypoints[self.RIGHT_EYE][:2], dtype=np.float32)
        left_ear = np.array(keypoints[self.LEFT_EAR][:2], dtype=np.float32)
        right_ear = np.array(keypoints[self.RIGHT_EAR][:2], dtype=np.float32)
        left_shoulder = np.array(keypoints[self.LEFT_SHOULDER][:2], dtype=np.float32)
        right_shoulder = np.array(keypoints[self.RIGHT_SHOULDER][:2], dtype=np.float32)
        
        # Tính các điểm trung bình
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_eye = (left_eye + right_eye) / 2
        mid_ear = (left_ear + right_ear) / 2
        
        # SỬ DỤNG SHOULDER_WIDTH thay vì body_height
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_width < 10:  # Quá nhỏ, không đáng tin cậy
            return None
        
        # Body reference ước tính từ shoulder width
        # Vai thường chiếm khoảng 40% chiều cao phần trên cơ thể
        body_reference = shoulder_width * 1.5
        
        return {
            'nose': nose,
            'mid_eye': mid_eye,
            'mid_ear': mid_ear,
            'mid_shoulder': mid_shoulder,
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'shoulder_width': shoulder_width,
            'body_reference': body_reference
        }
    
    def _calculate_features(self, points):
        """
        Tính toán các features phát hiện ngủ gật
        
        Returns:
            dict với các features và giá trị
        """
        nose = points['nose']
        mid_eye = points['mid_eye']
        mid_shoulder = points['mid_shoulder']
        shoulder_width = points['shoulder_width']
        body_ref = points['body_reference']
        
        features = {}
        
        # Feature 1: Head down ratio
        # Bình thường: nose_y < shoulder_y (đầu ở trên vai)
        # Cúi đầu: nose_y >= shoulder_y hoặc gần bằng
        vertical_dist = mid_shoulder[1] - nose[1]  # Dương = đầu ở trên
        features['head_down_ratio'] = vertical_dist / body_ref
        
        # Feature 2: Head tilt (đầu nghiêng sang bên)
        # Nếu ngủ gật, đầu có thể nghiêng
        horizontal_offset = abs(nose[0] - mid_shoulder[0])
        features['head_tilt'] = horizontal_offset / shoulder_width
        
        # Feature 3: Eye-shoulder distance 
        # Khi cúi đầu, mắt gần vai hơn
        eye_shoulder_dist = mid_shoulder[1] - mid_eye[1]
        features['eye_shoulder_ratio'] = eye_shoulder_dist / body_ref
        
        # Feature 4: Nose below eye line
        # Khi cúi đầu sâu, mũi có thể xuống dưới đường mắt nhiều
        nose_eye_dist = mid_eye[1] - nose[1]  # Dương = mũi ở dưới mắt
        features['nose_eye_ratio'] = nose_eye_dist / body_ref
        
        return features
    
    def _is_sleeping_pose(self, features):
        """
        Xác định xem pose hiện tại có phải là ngủ gật không
        
        Returns:
            (bool, float): (is_sleeping_pose, confidence_score)
        """
        head_down = features['head_down_ratio'] < self.head_down_threshold
        head_tilted = features['head_tilt'] > self.head_tilt_threshold
        
        # Scoring
        score = 0.0
        
        # Head down là điều kiện chính
        if head_down:
            score += 0.5
            # Càng cúi nhiều, score càng cao
            down_amount = self.head_down_threshold - features['head_down_ratio']
            score += min(0.2, down_amount * 2)
        
        # Head tilt là điều kiện phụ
        if head_tilted:
            score += 0.15
        
        # Eye gần shoulder 
        if features['eye_shoulder_ratio'] < 0.15:
            score += 0.15
        
        is_pose = score >= 0.5
        return is_pose, score
    
    def _calculate_motion(self, person_id):
        """
        Tính lượng motion từ keypoint history
        
        Returns:
            float: motion score (0 = không di chuyển)
        """
        history = self.keypoint_history[person_id]
        
        if len(history) < 3:
            return 1.0  # Không đủ data
        
        motion_scores = []
        
        for i in range(1, min(len(history), 10)):  # Chỉ xét 10 frames gần nhất
            prev_points = history[-(i+1)]
            curr_points = history[-i]
            
            if prev_points is None or curr_points is None:
                continue
            
            # So sánh vị trí nose và shoulder
            nose_motion = np.linalg.norm(curr_points['nose'] - prev_points['nose'])
            shoulder_motion = np.linalg.norm(curr_points['mid_shoulder'] - prev_points['mid_shoulder'])
            
            # Normalize
            ref = curr_points['body_reference']
            motion = (nose_motion + shoulder_motion) / (2 * ref)
            motion_scores.append(motion)
        
        if not motion_scores:
            return 1.0
        
        return np.mean(motion_scores)
    
    def _smooth_prediction(self, person_id, is_sleeping_pose, pose_score):
        """
        Làm mượt prediction bằng sliding window
        """
        self.feature_history[person_id].append((is_sleeping_pose, pose_score))
        
        if len(self.feature_history[person_id]) < 3:
            return is_sleeping_pose, pose_score
        
        # Voting từ window
        poses = [p[0] for p in self.feature_history[person_id]]
        scores = [p[1] for p in self.feature_history[person_id]]
        
        smoothed_pose = sum(poses) > len(poses) / 2
        smoothed_score = np.mean(scores)
        
        return smoothed_pose, smoothed_score
    
    def predict_keypoint(self, keypoint_result, mot_result, frame_id=0):
        """
        Dự đoán hành động ngủ gục từ keypoint và MOT results
        
        Args:
            keypoint_result: dict với 'keypoint' key, shape (N, 17, 3)
            mot_result: dict với 'boxes' key
            frame_id: current frame number
        
        Returns:
            list of (person_id, {'class': 0/1, 'score': float})
            class 0 = sleeping, class 1 = normal
        """
        results = []
        current_time = frame_id / self.fps
        
        keypoints = keypoint_result.get('keypoint', [])
        mot_boxes = mot_result.get('boxes', [])
        
        if len(keypoints) == 0 or len(mot_boxes) == 0:
            return results
        
        for idx in range(min(len(keypoints), len(mot_boxes))):
            kpts = keypoints[idx]
            person_id = int(mot_boxes[idx][0])
            
            # Extract key points với confidence filter
            points = self._extract_key_points(kpts)
            
            if points is None:
                # Sử dụng cache nếu có
                if person_id in self.result_cache:
                    cls, score, life = self.result_cache[person_id]
                    life -= 1
                    if life > 0:
                        self.result_cache[person_id] = (cls, score * 0.98, life)
                        results.append((person_id, {'class': cls, 'score': score * 0.98}))
                    else:
                        del self.result_cache[person_id]
                        results.append((person_id, {'class': 1, 'score': 0.5}))
                else:
                    results.append((person_id, {'class': 1, 'score': 0.5}))
                continue
            
            # Lưu vào history
            self.keypoint_history[person_id].append(points)
            
            # Tính features
            features = self._calculate_features(points)
            
            # Kiểm tra pose
            is_sleeping_pose, pose_score = self._is_sleeping_pose(features)
            
            # Smooth prediction
            is_sleeping_pose, pose_score = self._smooth_prediction(
                person_id, is_sleeping_pose, pose_score
            )
            
            # Tính motion
            motion_score = self._calculate_motion(person_id)
            is_still = motion_score < self.motion_threshold
            
            # Logic xác định ngủ gục
            if is_sleeping_pose and is_still:
                if person_id not in self.head_down_start:
                    self.head_down_start[person_id] = current_time
                
                duration = current_time - self.head_down_start[person_id]
                
                if duration >= self.duration_threshold:
                    # XÁC NHẬN ĐANG NGỦ
                    self.sleeping_status[person_id] = True
                    
                    # Score tăng theo thời gian
                    final_score = min(0.98, pose_score + 0.1 * (duration - self.duration_threshold))
                    self.result_cache[person_id] = (0, final_score, self.display_frames)
                    results.append((person_id, {'class': 0, 'score': final_score}))
                else:
                    # Đang theo dõi, chưa đủ thời gian
                    progress = duration / self.duration_threshold
                    intermediate_score = 0.3 + 0.3 * progress
                    results.append((person_id, {'class': 1, 'score': intermediate_score}))
            
            elif is_sleeping_pose and not is_still:
                # Có pose ngủ nhưng đang di chuyển (có thể đang gật gù)
                if person_id in self.head_down_start:
                    # Reset timer nhưng không xóa hoàn toàn
                    self.head_down_start[person_id] = current_time - self.duration_threshold * 0.3
                results.append((person_id, {'class': 1, 'score': pose_score * 0.5}))
            
            else:
                # Không phải pose ngủ
                if person_id in self.head_down_start:
                    del self.head_down_start[person_id]
                self.sleeping_status[person_id] = False
                
                # Giảm dần cache
                if person_id in self.result_cache:
                    cls, score, life = self.result_cache[person_id]
                    life -= 3
                    if life > 0:
                        self.result_cache[person_id] = (cls, score * 0.9, life)
                        results.append((person_id, {'class': cls, 'score': score * 0.9}))
                    else:
                        del self.result_cache[person_id]
                        results.append((person_id, {'class': 1, 'score': 0.5}))
                else:
                    results.append((person_id, {'class': 1, 'score': 0.5}))
        
        return results
    
    def get_sleeping_persons(self):
        """Trả về danh sách person IDs đang ngủ"""
        return [pid for pid, is_sleeping in self.sleeping_status.items() if is_sleeping]
    
    def get_stats(self):
        """Trả về thống kê tracking"""
        return {
            'total_tracked': len(self.keypoint_history),
            'sleeping_count': len(self.get_sleeping_persons()),
            'sleeping_ids': self.get_sleeping_persons()
        }
    
    def reset(self):
        """Reset tất cả state"""
        self.keypoint_history.clear()
        self.feature_history.clear()
        self.head_down_start.clear()
        self.sleeping_status.clear()
        self.result_cache.clear()


# Backward compatibility - alias
SleepingActionRecognizer = SleepingActionRecognizerV2


def visualize_sleeping_v2(frame, results, mot_boxes, show_debug=False):
    """
    Vẽ kết quả sleeping detection lên frame
    
    Args:
        frame: numpy array BGR image
        results: list of (person_id, {'class': 0/1, 'score': float})
        mot_boxes: MOT result boxes
        show_debug: Hiển thị thêm debug info
    
    Returns:
        frame với annotations
    """
    import cv2
    
    result_dict = {pid: res for pid, res in results}
    
    for box in mot_boxes:
        person_id = int(box[0])
        
        # Parse bbox
        if len(box) >= 6:
            x1, y1, x2, y2 = map(int, box[2:6])
        else:
            x1, y1, x2, y2 = map(int, box[1:5])
        
        if person_id in result_dict:
            cls = result_dict[person_id]['class']
            score = result_dict[person_id]['score']
            
            if cls == 0:  # Sleeping
                color = (0, 0, 255)  # Red
                label = f"SLEEPING {score:.0%}"
                thickness = 3
            else:  # Normal
                if score > 0.6:  # Đang theo dõi
                    color = (0, 165, 255)  # Orange
                    label = f"Monitoring {score:.0%}"
                else:
                    color = (0, 255, 0)  # Green
                    label = f"Normal"
                thickness = 2
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw person ID
            cv2.putText(frame, f"ID:{person_id}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


# Test function
if __name__ == "__main__":
    print("SleepingActionRecognizerV2 - Test")
    print("="*50)
    
    recognizer = SleepingActionRecognizerV2(
        head_down_threshold=0.10,
        duration_threshold=2.5,
        min_confidence=0.4
    )
    
    # Tạo test data
    test_keypoints = np.array([
        [100, 80, 0.9],   # nose
        [90, 70, 0.8],    # left_eye
        [110, 70, 0.8],   # right_eye
        [80, 75, 0.7],    # left_ear
        [120, 75, 0.7],   # right_ear
        [70, 120, 0.85],  # left_shoulder
        [130, 120, 0.85], # right_shoulder
        [60, 160, 0.6],   # left_elbow
        [140, 160, 0.6],  # right_elbow
        [50, 200, 0.5],   # left_wrist
        [150, 200, 0.5],  # right_wrist
        [80, 200, 0.6],   # left_hip
        [120, 200, 0.6],  # right_hip
        [75, 280, 0.5],   # left_knee
        [125, 280, 0.5],  # right_knee
        [70, 350, 0.4],   # left_ankle
        [130, 350, 0.4],  # right_ankle
    ])
    
    points = recognizer._extract_key_points(test_keypoints)
    if points:
        print(f"Shoulder width: {points['shoulder_width']:.2f}")
        print(f"Body reference: {points['body_reference']:.2f}")
        
        features = recognizer._calculate_features(points)
        print(f"\nFeatures:")
        for k, v in features.items():
            print(f"  {k}: {v:.3f}")
        
        is_sleeping, score = recognizer._is_sleeping_pose(features)
        print(f"\nIs sleeping pose: {is_sleeping}, Score: {score:.3f}")
    else:
        print("Failed to extract keypoints")
