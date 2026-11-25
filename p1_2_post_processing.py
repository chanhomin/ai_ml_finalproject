import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import butter, filtfilt
from scipy.ndimage import minimum_filter1d
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 후처리 설정값 (실험으로 결정) ----
FPS = 30.0                # 영상 프레임 속도
JOINT_CUTOFF_HZ = 6.0     # 관절 떨림 제거용 
GROUND_CUTOFF_HZ = 3.0    # 바닥 높이 곡선용 
GROUND_MIN_WINDOW = 5     # 바닥 하한선을 구할 프레임 수
FOOT_JOINTS = [29, 30, 31, 32]   # 왼/오른 발뒤꿈치, 왼/오른 발끝


def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)


def create_height_corrected_target(raw_seq):
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 모든 프레임에서 발의 y값 중 최소값 → ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    
    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    # 땅 아래(-)로 떨어지지 않게 clamp
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)

    return seq


def lowpass_filter(data, cutoff_hz, fps=FPS, order=2):
    """
    시간축(axis=0) 방향 저역 통과 필터.
    cutoff_hz보다 빠르게 변하는 성분(떨림)을 줄인다.
    """
    b, a = butter(order, cutoff_hz / (fps / 2))
    return filtfilt(b, a, data, axis=0)


def create_enhanced_target(raw_seq):
    """
    raw_seq: (T,33,3)
    Return enhanced target (T,33,3)

    1) 모든 관절의 떨림 제거
    2) 발 4점 기준의 부드러운 바닥 높이로 보정 (clamp 없음)
    """
    seq = raw_seq.astype(np.float64).copy()

    # 1) 관절 떨림 제거
    seq = lowpass_filter(seq, JOINT_CUTOFF_HZ)

    # 2) 프레임별로 발뒤꿈치/발끝 4점 중 가장 낮은 높이
    lowest = seq[:, FOOT_JOINTS, 1].min(axis=1)          # (T,)

    # 3) 주변 프레임 중 최저값 → 발 아래에 놓이는 하한선
    envelope = minimum_filter1d(lowest, size=GROUND_MIN_WINDOW, mode="nearest")

    # 4) 하한선을 부드럽게 → 최종 바닥 높이 곡선
    ground = lowpass_filter(envelope, GROUND_CUTOFF_HZ)

    # 5) 바닥이 y=0이 되도록 몸 전체를 이동 (강제로 누르지 않음)
    seq[:, :, 1] -= ground[:, None]

    return seq


def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    Export Unity-compatible CSV.
    """

    T, n_landmarks, _ = motion.shape

    rows = []

    for t in range(T):
        for landmark in range(n_landmarks):

            x, y, z = motion[t, landmark]

            rows.append([
                t,
                landmark,
                float(x),
                float(y),
                float(z),
                1.0
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility"
        ]
    )

    # 문자열 경로 처리
    output_path = os.path.abspath(output_path)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Unity CSV saved: {output_path}")
    print(f"Frames: {T}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Total rows: {len(df)}")

def main():
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        target_seq = create_height_corrected_target(raw_seq)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

        OUTPUT_PATH = (
                os.path.join(OUT_DIR,
                f"{base}_target.csv")
            )
        export_unity_csv(target_seq,OUTPUT_PATH)

        # ---------------------------------------
        # 3) Create enhanced target (후처리)
        # ---------------------------------------
        enhanced_seq = create_enhanced_target(raw_seq)
        enhanced_out_path = os.path.join(OUT_DIR, f"{base}_enhanced.npy")
        np.save(enhanced_out_path, enhanced_seq)
        print(f"Saved enhanced → {enhanced_out_path}")

        export_unity_csv(
            enhanced_seq,
            os.path.join(OUT_DIR, f"{base}_enhanced.csv")
        )

    print("\nDone processing all keypoint files.")

    if not paths:
        return

    # 시각화: 기존 후처리(왼쪽) vs 개선된 후처리(오른쪽)
    target = np.load(target_out_path)
    enhanced = np.load(enhanced_out_path)
    print(target.shape, enhanced.shape)

    vis = PoseViser(fps=30)
    vis.server.scene.set_up_direction("+y")   # 데이터가 y축이 위쪽이므로 카메라도 맞춤
    vis.play_two_sequences(target, enhanced, offset=1.5)


if __name__ == "__main__":
    main()
