import cv2
import numpy as np
import math
import argparse
import os
from collections import deque


class LaneLineSpeedEstimator:
    def __init__(self, video_path, output_name="output/result_stable.mp4"):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到影片檔案: {video_path}")

        self.cap = cv2.VideoCapture(video_path)

        # FPS 設定
        file_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if 24.0 < file_fps < 26.0:
            self.fps = 24.98
        else:
            self.fps = file_fps
        self.dt = 1.0 / self.fps

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 參數設定：增加穩定性
        self.N = 15  # 平滑窗口加大到 15
        self.speed_history = []
        self.frame_count = 0
        self.smooth_speed = 0.0

        # 歷史軌跡佇列 (用來儲存前幾幀的車道線位置)
        # 格式: deque 裡存 (frame_index, list_of_lanes)
        self.history_len = 6
        self.lane_history = deque(maxlen=self.history_len)

        # 全域平均比例尺 (避免比例尺忽大忽小)
        self.global_avg_height_px = 0.0
        self.valid_height_counts = 0

        self.REAL_WORLD_LANE_LENGTH = 10.0

        # 輸出設定
        os.makedirs("output", exist_ok=True)
        self.bev_width = 600
        self.bev_height = 800
        self.out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                   (self.frame_width, self.frame_height))
        self.out_binary = cv2.VideoWriter("output/result_binary.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                          (self.bev_width, self.bev_height))

        # 透視變換設定
        self.src_pts = np.float32([[1457, 1129], [1700, 1126], [1930, 1295], [1165, 1286]])
        self.dst_pts = np.float32(
            [[0, 0], [self.bev_width - 1, 0], [self.bev_width - 1, self.bev_height - 1], [0, self.bev_height - 1]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def _preprocess(self, frame):
        # ... (保持不變) ...
        frame_bev = cv2.warpPerspective(frame, self.M, (self.bev_width, self.bev_height))
        gray = cv2.cvtColor(frame_bev, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
        return frame_bev, binary_clean

    def _get_lane_segments(self, binary_clean):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        lane_segments = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            aspect_ratio = h / float(w) if w > 0 else 0
            if 100 < area < 5000:
                if aspect_ratio > 1.5:
                    lane_segments.append({'centroid': (cx, cy), 'bbox': (x, y, w, h), 'height_px': h})
        return lane_segments

    def _compute_speed_from_lanes(self, current_lanes):

        # 1. 將現在的車道線存入歷史紀錄
        self.lane_history.append(current_lanes)

        # 2. 如果歷史資料還不夠 (少於 5 幀)，先跳過不計算
        if len(self.lane_history) < self.history_len:
            return

        # 3. 取出「現在」與「5 幀前」的資料
        # history[-1] 是最新, history[0] 是最舊 (5幀前)
        past_lanes = self.lane_history[0]
        frame_diff = self.history_len - 1  # 實際間隔幀數

        frame_speeds = []

        for curr in current_lanes:
            cx, cy = curr['centroid']
            h_px = curr['height_px']

            # --- [穩定比例尺更新] ---
            # 使用移動平均更新全域線段高度，避免比例尺亂跳
            if self.valid_height_counts == 0:
                self.global_avg_height_px = h_px
            else:
                self.global_avg_height_px = 0.95 * self.global_avg_height_px + 0.05 * h_px
            self.valid_height_counts += 1

            # 尋找 5 幀前的對應線段
            best_match = None
            min_dist = 1e9

            for prev in past_lanes:
                px, py = prev['centroid']
                dist = math.hypot(cx - px, cy - py)

                # 假設車速很快，5幀可能跑了 200-300 pixel，設寬鬆閾值
                if dist < 400 and dist < min_dist:
                    min_dist = dist
                    best_match = prev

            if best_match:
                # 計算 Y 軸位移 (5 幀的總位移)
                pixel_displacement = abs(cy - best_match['centroid'][1])

                # 使用穩定的全域比例尺
                if self.global_avg_height_px > 0 and pixel_displacement > 0:
                    scale = self.REAL_WORLD_LANE_LENGTH / self.global_avg_height_px

                    distance_m = pixel_displacement * scale

                    # 時間 = dt * 間隔幀數
                    time_elapsed = self.dt * frame_diff

                    v = (distance_m / time_elapsed) * 3.6

                    if 5 < v < 200:
                        frame_speeds.append(v)

        if frame_speeds:
            avg_v = sum(frame_speeds) / len(frame_speeds)

            self.speed_history.append(avg_v)
            if len(self.speed_history) > self.N:
                self.speed_history.pop(0)

            self.smooth_speed = sum(self.speed_history) / len(self.speed_history)

            print(f"Frame {self.frame_count}: Speed={self.smooth_speed:.2f} km/h (Smoothed)")

    def _draw_results(self, frame, frame_bev, lanes):
        # ... (保持不變) ...
        if self.smooth_speed > 0:
            text = f"Speed: {self.smooth_speed:.1f} km/h"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        vis_bev = cv2.cvtColor(frame_bev, cv2.COLOR_GRAY2BGR) if len(frame_bev.shape) == 2 else frame_bev.copy()
        for lane in lanes:
            x, y, w, h = lane['bbox']
            cv2.rectangle(vis_bev, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = int(lane['centroid'][0]), int(lane['centroid'][1])
            cv2.circle(vis_bev, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Original", cv2.resize(frame, (800, 450)))
        cv2.imshow("Lane Tracking (BEV)", vis_bev)
        return vis_bev

    def run(self):
        # ... (保持不變) ...
        print(f"開始執行穩定版測速... (FPS={self.fps})")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            self.frame_count += 1
            frame_bev, binary_clean = self._preprocess(frame)
            current_lanes = self._get_lane_segments(binary_clean)
            self._compute_speed_from_lanes(current_lanes)
            vis_bev = self._draw_results(frame, frame_bev, current_lanes)
            self.out.write(frame)
            self.out_binary.write(vis_bev)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        self.cap.release()
        self.out.release()
        self.out_binary.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test1_mute.mp4")
    parser.add_argument("--output", type=str, default="output/result_stable.mp4")
    args = parser.parse_args()
    estimator = LaneLineSpeedEstimator(args.video, args.output)
    estimator.run()
