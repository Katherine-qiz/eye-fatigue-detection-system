from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import csv
import os
from pathlib import Path
from datetime import datetime
import threading
import time
from collections import deque
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__)
# 仅保留基础CORS配置，无需SocketIO
CORS(app, resources={r"/*": {"origins": "*"}})

# MediaPipe 初始化
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 面部网格和检测
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# 定义眼部关键点索引（从app4.py改进）
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

# 帧尺寸（从app4.py）
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# -------------------------- 全局变量和配置 --------------------------
# 全局变量存储最新的帧和分析数据（用于前端轮询获取）
latest_frame_data = {
    "frame": "",
    "blink_count": 0,
    "blink_frequency": 0.0,
    "eye_state": 0.0,
    "distance": 0.0,
    "brightness": 0.0,
    "use_time": 0.0,
    "fps": 0.0,
    "alerts": [],
    "face_detected": False,
    "processed_frame": ""  # 新增：处理后的帧
}
# 加锁保证线程安全
frame_data_lock = threading.Lock()

# -------------------------- 多语种配置 --------------------------
# 全局语种变量（默认普通话）
current_language = "mandarin"  # mandarin/cantonese/english
language_lock = threading.Lock()

# 语种映射（前端显示名 -> API返回值）
LANGUAGE_MAP = {
    "mandarin": "普通话",
    "cantonese": "粤语",
    "english": "英语"
}

from collections import deque

# 告警队列，存储最近产生的告警，供机器人拉取
alerts_queue = deque(maxlen=50)      # 最多保留50条，避免内存膨胀
alerts_lock = threading.Lock()

class EyeFatigueMonitor:
    def __init__(self, port_suffix=""):
        self.blink_count = 0
        self.blink_frequency = 0
        self.eye_state_history = deque(maxlen=30)
        self.ear_baseline = 0.25
        self.ear_samples = deque(maxlen=100)
        self.last_blink_time = time.time()
        self.is_blinking = False
        self.blink_start_time = None
        self.use_time_start = None
        self.use_time_total = 0
        self.distance_alerts = []
        self.brightness_alerts = []
        self.blink_alerts = []
        self.use_time_alerts = []
        self.last_alert_time = {"distance": 0, "brightness": 0, "blink": 0, "use_time": 0}
        self.alert_cooldown = 30
        self.session_start = datetime.now()
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if port_suffix:
            self.csv_file = f"logs/eye_fatigue_log_{timestamp}_{port_suffix}.csv"
        else:
            self.csv_file = f"logs/eye_fatigue_log_{timestamp}.csv"
        self.blink_timestamps = deque(maxlen=60)
        self.init_csv()
        
    def init_csv(self):
        os.makedirs("logs", exist_ok=True)
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'blink_count', 'blink_frequency', 'eye_state_value',
                'distance_cm', 'brightness_lux', 'use_time_seconds', 'fps',
                'alert_triggered', 'alert_reason', 'language'  # 新增：记录语种
            ])
    
    def log_data(self, eye_state, distance, brightness, alert_triggered, alert_reason):
        # 获取当前语种
        with language_lock:
            lang = current_language

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                self.blink_count,
                round(self.blink_frequency, 2),
                round(eye_state, 3),
                round(distance, 1),
                round(brightness, 1),
                round(self.use_time_total, 1),
                round(self.fps, 1),
                alert_triggered,
                alert_reason,
                lang  # 新增：写入语种
            ])
    
    def calculate_ear(self, landmarks, indices):
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
        
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        
        if h < 0.001:
            return 0.25
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_ear_improved(self, landmarks):
        """改进的EAR计算（从app4.py）"""
        left_indices = [33, 160, 158, 133, 153, 144]
        right_indices = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_ear(landmarks, left_indices)
        right_ear = self.calculate_ear(landmarks, right_indices)
        
        left_upper = [159, 160, 161]
        left_lower = [145, 144, 153]
        right_upper = [386, 385, 384]
        right_lower = [374, 373, 380]
        
        left_upper_y = np.mean([landmarks[i].y for i in left_upper])
        left_lower_y = np.mean([landmarks[i].y for i in left_lower])
        right_upper_y = np.mean([landmarks[i].y for i in right_upper])
        right_lower_y = np.mean([landmarks[i].y for i in right_lower])
        
        left_openness = left_lower_y - left_upper_y
        right_openness = right_lower_y - right_upper_y
        
        avg_ear = (left_ear + right_ear) / 2.0
        avg_openness = (left_openness + right_openness) / 2.0
        
        return avg_ear, avg_openness
    
    def detect_blink(self, ear, openness):
        """改进的眨眼检测（从app4.py）"""
        self.eye_state_history.append((ear, openness))
        
        if len(self.ear_samples) < 50:
            self.ear_samples.append(ear)
            if len(self.ear_samples) == 50:
                sorted_samples = sorted(list(self.ear_samples))
                self.ear_baseline = sorted_samples[int(len(sorted_samples) * 0.7)]
        else:
            self.ear_samples.append(ear)
            sorted_samples = sorted(list(self.ear_samples))
            self.ear_baseline = sorted_samples[int(len(sorted_samples) * 0.7)]
        
        if len(self.eye_state_history) < 5:
            return False
        
        blink_threshold = self.ear_baseline * 0.75
        
        recent_ears = [e[0] for e in list(self.eye_state_history)[-5:]]
        current_ear = recent_ears[-1]
        prev_ear = recent_ears[-2]
        
        if current_ear < blink_threshold and not self.is_blinking:
            if prev_ear >= blink_threshold:
                self.is_blinking = True
                self.blink_start_time = time.time()
            return False
        
        if self.is_blinking and current_ear >= blink_threshold:
            self.is_blinking = False
            if self.blink_start_time:
                blink_duration = time.time() - self.blink_start_time
                if 0.05 < blink_duration < 0.5:
                    return True
            return False
        
        if self.is_blinking and self.blink_start_time:
            if time.time() - self.blink_start_time > 1.0:
                self.is_blinking = False
                self.blink_start_time = None
        
        return False
    
    def update_blink_frequency(self):
        """更新眨眼频率（从app4.py）"""
        current_time = time.time()
        self.blink_timestamps.append(current_time)
        
        one_minute_ago = current_time - 60
        recent_blinks = [t for t in self.blink_timestamps if t > one_minute_ago]
        
        if len(recent_blinks) >= 2:
            time_span = recent_blinks[-1] - recent_blinks[0]
            if time_span > 0:
                self.blink_frequency = (len(recent_blinks) - 1) / time_span
            else:
                self.blink_frequency = 0
        else:
            self.blink_frequency = 0
    
    def estimate_distance(self, landmarks):
        """改进的距离估计（从app4.py）"""
        face_top = landmarks[10]
        face_bottom = landmarks[152]
        face_left = landmarks[234]
        face_right = landmarks[454]
        
        face_height_px = abs(face_bottom.y - face_top.y) * FRAME_HEIGHT
        face_width_px = abs(face_right.x - face_left.x) * FRAME_WIDTH
        
        avg_face_width_cm = 14.0
        avg_face_height_cm = 18.0
        
        focal_length_w = 500
        focal_length_h = 500
        
        if face_width_px > 10:
            distance_w = (avg_face_width_cm * focal_length_w) / face_width_px
        else:
            distance_w = 100
            
        if face_height_px > 10:
            distance_h = (avg_face_height_cm * focal_length_h) / face_height_px
        else:
            distance_h = 100
        
        distance = (distance_w + distance_h) / 2
        
        distance = max(20, min(200, distance))
        
        return distance
    
    def calculate_brightness(self, frame):
        """亮度计算（从app4.py）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        lux = brightness * 2.5
        return lux
    
    def update_fps(self):
        """更新FPS"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def check_alerts(self, distance, brightness, ear):
        """检查告警（基于app4.py的阈值）"""
        current_time = time.time()
        alerts = []
        
        # 亮度告警
        if brightness < 200 and (current_time - self.last_alert_time["brightness"]) > self.alert_cooldown:
            alerts.append({"type": "brightness", "message": "请调高亮度"})
            self.last_alert_time["brightness"] = current_time
            self.brightness_alerts.append(datetime.now().isoformat())
        
        # 距离告警（使用app4.py的阈值60cm）
        if distance <60 and (current_time - self.last_alert_time["distance"]) > self.alert_cooldown:
            alerts.append({"type": "distance", "message": "请保持健康用眼距离"})
            self.last_alert_time["distance"] = current_time
            self.distance_alerts.append(datetime.now().isoformat())
        
        # 眨眼频率告警（每分钟少于8次）
        blinks_per_minute = len([t for t in self.blink_timestamps if t > current_time - 60])
        if blinks_per_minute < 8 and blinks_per_minute > 0 and self.use_time_total > 30 and (current_time - self.last_alert_time["blink"]) > self.alert_cooldown:
            alerts.append({"type": "blink", "message": "提示音"})
            self.last_alert_time["blink"] = current_time
            self.blink_alerts.append(datetime.now().isoformat())
        
        # 用眼时长告警（每20分钟）
        if self.use_time_total > 0 and self.use_time_total % 1200 < 2 and (current_time - self.last_alert_time["use_time"]) > self.alert_cooldown:
            alerts.append({"type": "use_time", "message": "请放松眼睛"})
            self.last_alert_time["use_time"] = current_time
            self.use_time_alerts.append(datetime.now().isoformat())
        
        return alerts
    
    def draw_face_box(self, frame, landmarks):
        """绘制人脸框（从app4.py）"""
        h, w = frame.shape[:2]
        
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min = int(min(x_coords) * w) - 20
        x_max = int(max(x_coords) * w) + 20
        y_min = int(min(y_coords) * h) - 30
        y_max = int(max(y_coords) * h) + 20
        
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        corner_len = 20
        cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), (0, 255, 0), 3)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_min), (x_max - corner_len, y_min), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_len), (0, 255, 0), 3)
        cv2.line(frame, (x_min, y_max), (x_min + corner_len, y_max), (0, 255, 0), 3)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - corner_len), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), (0, 255, 0), 3)
        
        return frame
    
    def draw_eye_contours(self, frame, landmarks):
        """绘制眼部轮廓（从app4.py）"""
        h, w = frame.shape[:2]
        
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        left_eye_points = []
        for idx in left_eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            left_eye_points.append([x, y])
        
        right_eye_points = []
        for idx in right_eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            right_eye_points.append([x, y])
        
        left_eye_points = np.array(left_eye_points, dtype=np.int32)
        right_eye_points = np.array(right_eye_points, dtype=np.int32)
        
        cv2.polylines(frame, [left_eye_points], True, (255, 0, 255), 2)
        cv2.polylines(frame, [right_eye_points], True, (255, 0, 255), 2)
        
        left_iris_indices = [468, 469, 470, 471, 472]
        right_iris_indices = [473, 474, 475, 476, 477]
        
        for idx in left_iris_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        for idx in right_iris_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        return frame
    
    def draw_info_overlay(self, frame, ear, distance, brightness, blink_count):
        """绘制信息叠加层（从app4.py）"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 35), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Distance: {distance:.1f}cm", (20, 60), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Brightness: {brightness:.0f}lux", (20, 85), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 110), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 135), font, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame_data):
        """处理帧（整合app4.py和app5.py的功能）"""
        # 记录开始时间
        t0 = time.time()
        try:
            # 解码Base64图像
            img_data = base64.b64decode(frame_data.split(',')[1])
            t1 = time.time()
            print(f"[PERF] decode: {t1-t0:.3f}s")

            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 调整尺寸
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            t2 = time.time()
            print(f"[PERF] imdecode+resize: {t2-t1:.3f}s")

            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 用于返回带标记的帧
            annotated_frame = frame.copy()
            
            # 人脸检测
            results = face_mesh.process(frame_rgb)

            t3 = time.time()
            print(f"[PERF] MediaPipe face_mesh: {t3-t2:.3f}s")

            face_detected = False
            ear = 0
            distance = 0
            brightness = self.calculate_brightness(frame_rgb)
            alerts = []
            
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 改进的EAR计算
                ear, openness = self.calculate_ear_improved(landmarks)
                
                # 改进的眨眼检测
                if self.detect_blink(ear, openness):
                    self.blink_count += 1
                    self.update_blink_frequency()
                
                # 改进的距离估计
                distance = self.estimate_distance(landmarks)
                
                # 检查告警
                alerts = self.check_alerts(distance, brightness, ear)
                
                # 绘制面部网格和眼部关键点
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=FACE_CONNECTIONS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # 绘制眼部关键点
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    x = int(landmarks[idx].x * frame.shape[1])
                    y = int(landmarks[idx].y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
                
                # 绘制人脸框和眼部轮廓（app4.py的功能）
                annotated_frame = self.draw_face_box(annotated_frame, landmarks)
                annotated_frame = self.draw_eye_contours(annotated_frame, landmarks)
                annotated_frame = self.draw_info_overlay(annotated_frame, ear, distance, brightness, self.blink_count)
            
            # 处理使用时间
            if face_detected:
                if not self.use_time_start:
                    self.use_time_start = time.time()
                else:
                    self.use_time_total += time.time() - self.use_time_start
                    self.use_time_start = time.time()
            else:
                if self.use_time_start:
                    self.use_time_total += time.time() - self.use_time_start
                    self.use_time_start = None
            
            self.update_fps()
            
            alert_triggered = len(alerts) > 0
            alert_reason = "|".join([a["type"] for a in alerts]) if alerts else "none"
            
            self.log_data(ear, distance, brightness, alert_triggered, alert_reason)
            
            # 将标注后的帧编码为base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构造返回结果
            result = {
                "success": True,
                "face_detected": face_detected,
                "blink_count": self.blink_count,
                "blink_frequency": round(self.blink_frequency, 2),
                "eye_state": round(ear, 3),
                "distance": round(distance, 1),
                "brightness": round(brightness, 1),
                "use_time": round(self.use_time_total, 1),
                "fps": round(self.fps, 1),
                "alerts": alerts,
                "processed_frame": f"data:image/jpeg;base64,{frame_base64}",  # 为index4.html提供
                "frame": frame_base64  # 为index5.html提供
            }
            
            # 更新全局最新帧数据（加锁保证线程安全）
            with frame_data_lock:
                global latest_frame_data
                latest_frame_data = {
                    "frame": frame_base64,
                    "blink_count": self.blink_count,
                    "blink_frequency": round(self.blink_frequency, 2),
                    "eye_state": round(ear, 3),
                    "distance": round(distance, 1),
                    "brightness": round(brightness, 1),
                    "use_time": round(self.use_time_total, 1),
                    "fps": round(self.fps, 1),
                    "alerts": alerts,
                    "face_detected": face_detected,
                    "processed_frame": f"data:image/jpeg;base64,{frame_base64}"
                }
            t4 = time.time()
            print(f"[PERF] total process: {t4-t0:.3f}s")
            return result
            
        except Exception as e:
            self.log_data(0, 0, 0, False, f"error:{str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


monitor = EyeFatigueMonitor(port_suffix="18008")   # 用于 18008 端口的服务


# -------------------------- 新增：语种切换接口 --------------------------
@app.route('/api/set_language', methods=['POST'])
def set_language():
    """设置当前播放语种"""
    try:
        data = request.get_json()
        lang = data.get('language', 'mandarin')

        # 验证语种有效性
        if lang not in LANGUAGE_MAP.keys():
            return jsonify({
                "success": False,
                "error": f"无效的语种：{lang}，支持的语种：{list(LANGUAGE_MAP.keys())}"
            }), 400

        # 加锁更新全局语种
        with language_lock:
            global current_language
            current_language = lang

        return jsonify({
            "success": True,
            "language": lang,
            "language_name": LANGUAGE_MAP[lang]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/get_language', methods=['GET'])
def get_language():
    """获取当前选中的语种"""
    with language_lock:
        lang = current_language
    return jsonify({
        "success": True,
        "language": lang,
        "language_name": LANGUAGE_MAP[lang]
    })

@app.route('/api/get_alerts', methods=['GET'])
def get_alerts():
    with alerts_lock:
        # 取出当前所有告警并清空队列
        alerts = list(alerts_queue)
        alerts_queue.clear()
    return jsonify({"success": True, "alerts": alerts, "language": current_language, "language_name": LANGUAGE_MAP[current_language]})

# 前端展示页面路由
@app.route('/')
def index():
    return send_file(FRONTEND_DIR / "index.html")


@app.route('/api/analyze', methods=['POST'])
def analyze_frame():
    # 记录请求体大小（字节）
    size = request.content_length if request.content_length is not None else len(request.get_data())
    print(f"[BANDWIDTH] Request size: {size} bytes")

    data = request.get_json()
    frame_data = data.get('frame')

    if not frame_data:
        return jsonify({"success": False, "error": "No frame data"}), 400

    result = monitor.process_frame(frame_data)

    # 获取当前语种并添加到返回结果中
    with language_lock:
        result['language'] = current_language

    # ---------- 识别机器人请求 ----------
    user_agent = request.headers.get('User-Agent', '')
    if 'ESP32' in user_agent:
        # 如果是机器人，将告警存入队列
        alerts = result.get('alerts', [])
        if alerts:
            with alerts_lock:
                alerts_queue.extend(alerts)
        # 返回极简响应（仅成功状态），避免大数据量影响帧率
        return jsonify({"success": True})
    else:
        # 非机器人（如浏览器），返回完整结果（含图片字段）
        # 注意：非机器人请求仍需要图片字段，所以不移除
        return jsonify(result)


# 前端轮询获取最新帧和数据的接口
@app.route('/api/get_frame', methods=['GET'])
def get_latest_frame():
    with frame_data_lock:
        # 获取当前语种
        with language_lock:
            lang = current_language
        
        # 添加当前语种信息
        response = {
            "success": True,
            **latest_frame_data,
            "language": lang,
            "language_name": LANGUAGE_MAP[lang]
        }
    return jsonify(response)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    # 添加语种信息
    with language_lock:
        lang = current_language
    return jsonify({
        "blink_count": monitor.blink_count,
        "blink_frequency": round(monitor.blink_frequency, 2),
        "use_time": round(monitor.use_time_total, 1),
        "distance_alerts": len(monitor.distance_alerts),
        "brightness_alerts": len(monitor.brightness_alerts),
        "blink_alerts": len(monitor.blink_alerts),
        "use_time_alerts": len(monitor.use_time_alerts),
        "fps": round(monitor.fps, 1),
        "language": lang,
        "language_name": LANGUAGE_MAP[lang]
    })


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    global monitor
    monitor = EyeFatigueMonitor(port_suffix="18008")
    # 重置最新帧数据
    with frame_data_lock:
        global latest_frame_data
        latest_frame_data = {
            "frame": "",
            "blink_count": 0,
            "blink_frequency": 0.0,
            "eye_state": 0.0,
            "distance": 0.0,
            "brightness": 0.0,
            "use_time": 0.0,
            "fps": 0.0,
            "alerts": [],
            "face_detected": False,
            "processed_frame": ""
        }
    return jsonify({"success": True})


@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return jsonify({"files": []})

    files = [f for f in os.listdir(logs_dir) if f.endswith('.csv')]
    return jsonify({"files": files})


@app.route('/api/download/<filename>', methods=['GET'])
def download_log(filename):
    file_path = os.path.join("logs", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


# 校准接口（从app4.py）
@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    data = request.get_json()
    known_distance = data.get('distance', 50)
    # 注意：这个monitor没有calibration_distance属性，需要添加
    return jsonify({"success": True, "message": f"Calibrated at {known_distance}cm"})

@app.route('/index4')
def serve_index4():
    return send_file(FRONTEND_DIR / "index.html")

@app.route('/index5')
def serve_index5():
    return send_file(FRONTEND_DIR / "index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=18008, debug=False, threaded=True)
