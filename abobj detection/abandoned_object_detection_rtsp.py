"""
============================================================
  ABANDONED OBJECT DETECTION SYSTEM
  Supports: RTSP Camera Feed / Video File / Webcam
  Built with: YOLOv8 + DeepSORT + OpenCV MOG2
============================================================

INSTALL DEPENDENCIES:
    pip install ultralytics deep_sort_realtime opencv-python

HOW TO RUN:

  1. RTSP Camera:
        python abandoned_object_detection_rtsp.py --source "rtsp://username:password@ip_address:port/stream"

  2. Video File:
        python abandoned_object_detection_rtsp.py --source "path/to/video.mp4"

  3. Webcam:
        python abandoned_object_detection_rtsp.py --source 0

  4. Save output video:
        python abandoned_object_detection_rtsp.py --source "rtsp://..." --save

  5. Run headless (no display window, just save):
        python abandoned_object_detection_rtsp.py --source "rtsp://..." --save --headless

RTSP URL FORMATS (common camera brands):
  Hikvision  : rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
  Dahua      : rtsp://admin:password@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0
  Axis       : rtsp://admin:password@192.168.1.64/axis-media/media.amp
  Generic    : rtsp://username:password@ip:554/stream1

CONTROLS (when window is open):
  Q or ESC   : Quit
  S          : Save screenshot
  P          : Pause / Resume
============================================================
"""

import cv2
import math
import time
import argparse
import os
import sys
import datetime
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ──────────────────────────────────────────────
#  CONFIGURATION — Edit these to tune behavior
# ──────────────────────────────────────────────
CONFIG = {
    # Detection
    'yolo_model'            : 'yolov8s.pt',   # yolov8n (fastest) | yolov8s | yolov8m (most accurate)
    'confidence_threshold'  : 0.25,            # lower = more sensitive

    # Object classes to monitor (COCO IDs)
    # 24=backpack, 25=umbrella, 26=handbag, 28=suitcase
    # 32=sports ball, 39=bottle, 41=cup, 45=bowl, 56=chair, 63=laptop, 67=phone
    'object_classes'        : [24, 25, 26, 27, 28, 32, 39, 41, 45, 56, 63, 67],
    'person_class'          : 0,

    # Abandonment logic
    'abandon_time_seconds'  : 15,   # seconds before flagging as abandoned
    'proximity_threshold'   : 120,  # px — nearby distance (passerby zone)
    'owner_threshold'       : 60,   # px — very close distance (actual owner)
    'stationary_threshold'  : 30,   # px — max movement to count as stationary
    'max_proximity_resets'  : 3,    # how many passersby can reset the timer

    # Tracking
    'max_age'               : 60,   # frames to keep a lost track alive

    # Background subtraction
    'bg_history'            : 100,  # low = catches static objects better
    'bg_threshold'          : 40,

    # RTSP reconnection
    'reconnect_attempts'    : 5,    # how many times to retry on stream loss
    'reconnect_delay'       : 3,    # seconds between retry attempts

    # Output
    'output_dir'            : 'output',
    'log_alerts'            : True,  # save alert log to CSV
}


# ──────────────────────────────────────────────
#  Object State
# ──────────────────────────────────────────────
class ObjectState:
    def __init__(self, track_id, bbox, class_id, fps, source='yolo'):
        self.track_id               = track_id
        self.class_id               = class_id
        self.bbox                   = bbox
        self.center                 = self._get_center(bbox)
        self.stationary_since_frame = None
        self.is_abandoned           = False
        self.locked_abandoned       = False
        self.fps                    = fps
        self.source                 = source
        self.last_seen_frame        = None
        self.last_seen_bbox         = bbox
        self.was_carried            = False
        self.frames_missing         = 0
        self.proximity_reset_count  = 0

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox, current_frame, stationary_thresh):
        new_center = self._get_center(bbox)
        dist = math.dist(self.center, new_center)
        if dist > stationary_thresh:
            if not self.locked_abandoned:
                self.stationary_since_frame = None
                self.is_abandoned = False
        else:
            if self.stationary_since_frame is None:
                self.stationary_since_frame = current_frame
        self.bbox            = bbox
        self.last_seen_bbox  = bbox
        self.center          = new_center
        self.last_seen_frame = current_frame
        self.frames_missing  = 0

    def get_stationary_seconds(self, current_frame):
        if self.stationary_since_frame is None:
            return 0
        return (current_frame - self.stationary_since_frame) / self.fps


# ──────────────────────────────────────────────
#  Main Detector
# ──────────────────────────────────────────────
class AbandonedObjectDetector:
    def __init__(self, config):
        self.config = config
        print("🔄 Loading YOLOv8 model...")
        self.model        = YOLO(config['yolo_model'])
        self.tracker      = DeepSort(max_age=config['max_age'])
        self.object_states  = {}
        self.bg_object_states = {}
        self.ghost_zones    = {}
        self.class_names    = self.model.names
        self.bg_id_counter  = 9000
        self.ghost_id_counter = 5000
        self.alert_log      = []   # list of alert events

        self.COLOR_NORMAL  = (0, 255, 0)
        self.COLOR_WARNING = (0, 165, 255)
        self.COLOR_DANGER  = (0, 0, 255)
        self.COLOR_PERSON  = (255, 200, 0)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config['bg_history'],
            varThreshold=config['bg_threshold'],
            detectShadows=False
        )
        print(f"✅ Model loaded: {config['yolo_model']}")

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _is_person_nearby(self, center, person_boxes, threshold=None):
        t = threshold or self.config['proximity_threshold']
        for pbox in person_boxes:
            if math.dist(center, self._get_center(pbox)) < t:
                return True
        return False

    def _get_bg_blobs(self, frame):
        mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 80000:
                x, y, w, h = cv2.boundingRect(cnt)
                blobs.append([x, y, x + w, y + h])
        return blobs

    def _match_blob(self, center, threshold=80):
        for tid, state in self.bg_object_states.items():
            if math.dist(center, state.center) < threshold:
                return tid
        return None

    def _draw_box(self, frame, bbox, label, color, is_abandoned, frame_idx):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_abandoned else 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if is_abandoned and (frame_idx // 10) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _log_alert(self, frame_idx, fps, label, bbox):
        timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        video_time = f"{frame_idx/fps:.1f}s"
        self.alert_log.append({
            'timestamp': timestamp,
            'video_time': video_time,
            'object': label,
            'bbox': bbox
        })

    def process_frame(self, frame, frame_idx, fps):
        cfg = self.config
        results = self.model(frame, verbose=False, conf=cfg['confidence_threshold'])[0]

        person_boxes      = []
        object_detections = []

        # ── YOLO detections ──────────────────────────────────
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            if cls_id == cfg['person_class']:
                person_boxes.append(bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_PERSON, 1)
                cv2.putText(frame, 'Person', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_PERSON, 1)
            elif cls_id in cfg['object_classes']:
                object_detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

        # ── DeepSORT ─────────────────────────────────────────
        tracks     = self.tracker.update_tracks(object_detections, frame=frame)
        active_ids = set()
        abandon_count = 0

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            active_ids.add(tid)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bbox   = [x1, y1, x2, y2]
            cls_id = track.det_class if track.det_class is not None else -1

            if tid not in self.object_states:
                self.object_states[tid] = ObjectState(tid, bbox, cls_id, fps)
                self.object_states[tid].stationary_since_frame = frame_idx
            else:
                self.object_states[tid].update(bbox, frame_idx, cfg['stationary_threshold'])

            state           = self.object_states[tid]
            stationary_secs = state.get_stationary_seconds(frame_idx)
            owner_close     = self._is_person_nearby(state.center, person_boxes, cfg['owner_threshold'])
            person_nearby   = self._is_person_nearby(state.center, person_boxes)

            if state.locked_abandoned:
                if owner_close:
                    state.locked_abandoned        = False
                    state.proximity_reset_count   = 0
                    state.stationary_since_frame  = frame_idx
                    stationary_secs = 0
                    is_abandoned = False
                else:
                    is_abandoned = True
                    abandon_count += 1
            else:
                if owner_close:
                    state.stationary_since_frame = frame_idx
                    stationary_secs = 0
                    state.proximity_reset_count  = 0
                elif person_nearby and not state.was_carried:
                    state.proximity_reset_count += 1
                    if state.proximity_reset_count <= cfg['max_proximity_resets']:
                        state.stationary_since_frame = frame_idx
                        stationary_secs = 0
                if person_nearby and state.was_carried:
                    state.stationary_since_frame = frame_idx
                    stationary_secs = 0

                is_abandoned = stationary_secs >= cfg['abandon_time_seconds'] and not owner_close
                if is_abandoned:
                    state.locked_abandoned = True
                    abandon_count += 1
                    obj_name = self.class_names.get(cls_id, 'Object')
                    self._log_alert(frame_idx, fps, obj_name, bbox)

            if person_nearby and stationary_secs == 0:
                state.was_carried = True

            state.is_abandoned = is_abandoned

            obj_name = self.class_names.get(cls_id, 'Object')
            if is_abandoned:
                color, status = self.COLOR_DANGER, f'ABANDONED! {stationary_secs:.1f}s'
            elif stationary_secs > cfg['abandon_time_seconds'] * 0.5:
                color, status = self.COLOR_WARNING, f'Stationary {stationary_secs:.1f}s'
            else:
                color, status = self.COLOR_NORMAL, 'Tracked'

            self._draw_box(frame, bbox, f'[{tid}] {obj_name} | {status}', color, is_abandoned, frame_idx)

        # ── Ghost Zones ───────────────────────────────────────
        for tid in list(self.object_states.keys()):
            if tid not in active_ids:
                state = self.object_states[tid]
                state.frames_missing = getattr(state, 'frames_missing', 0) + 1
                if state.last_seen_frame is not None and state.frames_missing < 5:
                    already = any(math.dist(g['center'], state.center) < 80 for g in self.ghost_zones.values())
                    if not already:
                        gid = self.ghost_id_counter
                        self.ghost_id_counter += 1
                        self.ghost_zones[gid] = {
                            'bbox': state.last_seen_bbox, 'center': state.center,
                            'stationary_since': state.stationary_since_frame or frame_idx,
                            'frames_missing': 0, 'class_id': state.class_id,
                            'locked': state.locked_abandoned,
                        }
                if state.frames_missing > cfg['max_age']:
                    del self.object_states[tid]

        for gid in list(self.ghost_zones.keys()):
            ghost = self.ghost_zones[gid]
            ghost['frames_missing'] += 1
            owner_close = self._is_person_nearby(ghost['center'], person_boxes, cfg['owner_threshold'])
            if owner_close:
                del self.ghost_zones[gid]
                continue
            if ghost['frames_missing'] > 90:
                del self.ghost_zones[gid]
                continue
            stationary_secs = (frame_idx - ghost['stationary_since']) / fps
            is_abandoned = stationary_secs >= cfg['abandon_time_seconds'] or ghost.get('locked', False)
            if is_abandoned:
                ghost['locked'] = True
                abandon_count += 1
                cls_name = self.class_names.get(ghost['class_id'], 'Object')
                label = f'[LOST] {cls_name} | ABANDONED! {stationary_secs:.1f}s'
                self._draw_box(frame, ghost['bbox'], label, self.COLOR_DANGER, True, frame_idx)
                cv2.circle(frame, ghost['center'], cfg['proximity_threshold'] // 3, (0, 0, 255), 1)
            elif stationary_secs > cfg['abandon_time_seconds'] * 0.5:
                cls_name = self.class_names.get(ghost['class_id'], 'Object')
                label = f'[LOST] {cls_name} | Stationary {stationary_secs:.1f}s'
                self._draw_box(frame, ghost['bbox'], label, self.COLOR_WARNING, False, frame_idx)

        # ── BG Subtraction Fallback ───────────────────────────
        bg_blobs      = self._get_bg_blobs(frame)
        matched_bg    = set()

        for blob_bbox in bg_blobs:
            bc = self._get_center(blob_bbox)
            if self._is_person_nearby(bc, person_boxes):
                continue
            if any(math.dist(bc, s.center) < 80 for s in self.object_states.values()):
                continue
            if any(math.dist(bc, g['center']) < 80 for g in self.ghost_zones.values()):
                continue
            existing = self._match_blob(bc)
            if existing is not None:
                tid = existing
                self.bg_object_states[tid].update(blob_bbox, frame_idx, cfg['stationary_threshold'])
                matched_bg.add(tid)
            else:
                tid = self.bg_id_counter
                self.bg_id_counter += 1
                s = ObjectState(tid, blob_bbox, -1, fps, source='bg')
                s.stationary_since_frame = frame_idx
                self.bg_object_states[tid] = s
                matched_bg.add(tid)

            state = self.bg_object_states[tid]
            stationary_secs = state.get_stationary_seconds(frame_idx)
            is_abandoned = stationary_secs >= cfg['abandon_time_seconds'] or state.locked_abandoned
            if is_abandoned:
                state.locked_abandoned = True
                abandon_count += 1
                label = f'[BG] Unattended | ABANDONED! {stationary_secs:.1f}s'
                self._draw_box(frame, blob_bbox, label, self.COLOR_DANGER, True, frame_idx)
                self._log_alert(frame_idx, fps, 'BG-Object', blob_bbox)
            elif stationary_secs > cfg['abandon_time_seconds'] * 0.5:
                label = f'[BG] Object | Stationary {stationary_secs:.1f}s'
                self._draw_box(frame, blob_bbox, label, self.COLOR_WARNING, False, frame_idx)

        for tid in list(self.bg_object_states.keys()):
            if tid not in matched_bg:
                s = self.bg_object_states[tid]
                if frame_idx - (s.stationary_since_frame or frame_idx) > cfg['max_age'] * 3:
                    del self.bg_object_states[tid]

        # ── HUD ──────────────────────────────────────────────
        h, w = frame.shape[:2]
        now  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cv2.rectangle(frame, (0, 0), (460, 85), (20, 20, 20), -1)
        cv2.putText(frame, 'Abandoned Object Detection System', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'{now}', (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(frame,
                    f'YOLO: {len(active_ids)}  BG: {len(matched_bg)}  Ghosts: {len(self.ghost_zones)}',
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        if abandon_count > 0:
            cv2.putText(frame, f'ABANDONED OBJECTS: {abandon_count}', (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (w//2 - 185, 8), (w//2 + 185, 50), (0, 0, 180), -1)
            cv2.putText(frame, '⚠  ABANDONED OBJECT ALERT  ⚠', (w//2 - 173, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Legend
        legend_items = [
            (self.COLOR_NORMAL,  'Tracked'),
            (self.COLOR_WARNING, 'Stationary (watch)'),
            (self.COLOR_DANGER,  'ABANDONED'),
            (self.COLOR_PERSON,  'Person'),
        ]
        ly = h - 80
        cv2.rectangle(frame, (0, ly - 8), (230, h), (20, 20, 20), -1)
        for i, (color, text) in enumerate(legend_items):
            y = ly + i * 18
            cv2.rectangle(frame, (8, y), (22, y + 12), color, -1)
            cv2.putText(frame, text, (28, y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        return frame, abandon_count


# ──────────────────────────────────────────────
#  Alert Logger
# ──────────────────────────────────────────────
def save_alert_log(alert_log, output_dir):
    if not alert_log:
        return
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"alerts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, 'w') as f:
        f.write("timestamp,video_time,object,bbox\n")
        for a in alert_log:
            f.write(f"{a['timestamp']},{a['video_time']},{a['object']},{a['bbox']}\n")
    print(f"📋 Alert log saved: {log_path}")


# ──────────────────────────────────────────────
#  RTSP Stream Handler with Reconnect
# ──────────────────────────────────────────────
def open_stream(source, cfg):
    """Open video stream with RTSP buffer optimization."""
    if isinstance(source, str) and source.startswith('rtsp'):
        # RTSP-specific settings for low latency
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize buffer lag
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return None
    return cap


def run(source, save=False, headless=False):
    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  ABANDONED OBJECT DETECTION SYSTEM")
    print(f"{'='*55}")
    print(f"  Source  : {source}")
    print(f"  Model   : {cfg['yolo_model']}")
    print(f"  Abandon : {cfg['abandon_time_seconds']}s")
    print(f"  Save    : {save}")
    print(f"  Headless: {headless}")
    print(f"{'='*55}\n")

    # ── Open stream ──────────────────────────────────
    cap = open_stream(source, cfg)
    if cap is None:
        print(f"❌ Cannot open source: {source}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Stream: {width}x{height} @ {fps:.1f}fps")

    # ── Output writer ─────────────────────────────────
    out = None
    if save:
        out_path = os.path.join(
            cfg['output_dir'],
            f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"💾 Saving to: {out_path}")

    detector    = AbandonedObjectDetector(cfg)
    frame_idx   = 0
    reconnects  = 0
    paused      = False
    total_alerts = 0

    print("\n🚀 Running... (Press Q or ESC to quit)\n")

    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key in [ord('q'), 27]:
                break
            if key == ord('p'):
                paused = False
                print("▶️  Resumed")
            continue

        ret, frame = cap.read()

        # ── Handle stream loss / reconnect ───────────────
        if not ret:
            if isinstance(source, str) and source.startswith('rtsp'):
                reconnects += 1
                if reconnects > cfg['reconnect_attempts']:
                    print("❌ Max reconnect attempts reached. Exiting.")
                    break
                print(f"⚠️  Stream lost. Reconnecting ({reconnects}/{cfg['reconnect_attempts']})...")
                cap.release()
                time.sleep(cfg['reconnect_delay'])
                cap = open_stream(source, cfg)
                if cap is None:
                    print("❌ Reconnect failed.")
                    break
                continue
            else:
                print("✅ End of video.")
                break

        reconnects = 0  # reset on successful frame

        # ── Process ───────────────────────────────────────
        processed, abandon_count = detector.process_frame(frame, frame_idx, fps)
        total_alerts += abandon_count

        if out:
            out.write(processed)

        # ── Display ───────────────────────────────────────
        if not headless:
            cv2.imshow('Abandoned Object Detection  [Q=Quit | P=Pause | S=Screenshot]', processed)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:       # Q or ESC — quit
                print("\n👋 Quit by user.")
                break
            elif key == ord('s'):           # S — screenshot
                shot_path = os.path.join(cfg['output_dir'], f"screenshot_{frame_idx}.jpg")
                cv2.imwrite(shot_path, processed)
                print(f"📸 Screenshot saved: {shot_path}")
            elif key == ord('p'):           # P — pause
                paused = True
                print("⏸️  Paused (press P to resume)")

        frame_idx += 1

        # Print status every 100 frames
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} | Time {frame_idx/fps:.1f}s | Total alert frames: {total_alerts}")

    # ── Cleanup ───────────────────────────────────────
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Save alert log
    if cfg['log_alerts']:
        save_alert_log(detector.alert_log, cfg['output_dir'])

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Alert events     : {len(detector.alert_log)}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Abandoned Object Detection System')

    parser.add_argument('--source',
        type=str, default='0',
        help='Source: RTSP URL | video file path | 0 for webcam')

    parser.add_argument('--save',
        action='store_true',
        help='Save output video to output/ folder')

    parser.add_argument('--headless',
        action='store_true',
        help='Run without display window (useful for servers)')

    parser.add_argument('--abandon-time',
        type=int, default=None,
        help='Override abandon time in seconds (default: 15)')

    parser.add_argument('--confidence',
        type=float, default=None,
        help='Override confidence threshold (default: 0.25)')

    parser.add_argument('--model',
        type=str, default=None,
        help='Override YOLO model (yolov8n / yolov8s / yolov8m)')

    args = parser.parse_args()

    # Apply CLI overrides
    if args.abandon_time : CONFIG['abandon_time_seconds'] = args.abandon_time
    if args.confidence   : CONFIG['confidence_threshold'] = args.confidence
    if args.model        : CONFIG['yolo_model']           = args.model

    # Handle webcam source
    source = args.source
    if source.isdigit():
        source = int(source)

    run(source=source, save=args.save, headless=args.headless)
