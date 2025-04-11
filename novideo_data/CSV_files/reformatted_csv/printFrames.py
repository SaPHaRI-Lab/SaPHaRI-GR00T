import cv2

video_path = "demo_data/robot_sim.PickNPlace/videos/chunk-000/observation.images.ego_view/episode_000004.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# 0 = 659
# 1 = 230
# 2 = 230
# 3 = 543