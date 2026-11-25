# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
import re
from yt_dlp import YoutubeDL
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def download_youtube(url):
    with YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    name = re.sub(r'[\\/:*?"<>|\s]+', "_", info["title"]).strip("_")
    filepath = os.path.join(RAW_VIDEO_DIR, f"{name}.mp4")

    ydl_opts = {
        "format": "bv*[ext=mp4][vcodec^=avc1][height<=1080]/b[ext=mp4]/b",
        "outtmpl": filepath,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          
                dynamic_ncols=False,  
                disable=True
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    # url_list=["https://youtube.com/shorts/fNvA0dU5fZM?si=FFBbj_IprAQ4S_fs",
    # "https://youtube.com/shorts/rCECyHdKGnA?si=CQ_qbTmh2p0k3oJ3",
    # "https://youtube.com/shorts/GX6Tl9KtQgE?si=hNApdx7ZNOY_U0su",
    # 'https://youtube.com/shorts/Xuzazm0VAik?si=2ugZmNk6XtKHCkxe',
    # "https://youtube.com/shorts/Z5IKezdRTKU?si=n7hRZHvkP2Z809hy",
    # "https://youtube.com/shorts/-XgnHvuScP8?si=yxnOz7JfpsjHnv7q"]
    # url_list = ["https://youtube.com/shorts/vl8sBA0oIYU?si=4qZn3gt-5u2hs2YY",
    #             "https://youtube.com/shorts/xXNFSn_GU7Y?si=xh9T-XFXJxbsSA5I",
    #             "https://youtube.com/shorts/cakv_6_PTBA?si=gqxtP8QKsfpJGsoY",
    #             "https://youtube.com/shorts/N62KmR7zN0A?si=vstkCnOmTsC60D1x",
    #             "https://youtube.com/shorts/3tvVRaFbKKE?si=afZR4ymGQJWGhvRQ"
    #             ]
    url_list=["https://www.youtube.com/shorts/jNrS0MjuSVQ"]
    for url in url_list:
        video_path, name = download_youtube(url)
        extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
        # Optional : Diverse Types of Data

if __name__ == "__main__":
    main()
