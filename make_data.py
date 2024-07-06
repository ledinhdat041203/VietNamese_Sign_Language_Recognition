import cv2
import mediapipe as mp
import pandas as pd
from imutils.video import VideoStream
import os
from datetime import datetime

OUTPUT_FILE = "dataset.txt"
TOTAL_POSE_LANDMARKS = 33 
TOTAL_HAND_LANDMARKS = 21  
TOTAL_HANDS = 2   
NUM_FRAME_PROCESS = 25 # So frame muon xu ly
NOSE_POSITION = 0


def fixed_num_frame(lst_frame, num_frame):
    total_frame = len(lst_frame)
    
    num_step = max(total_frame // (num_frame - 1), 1)

    new_lst = []

    for i in range(num_frame):
        idx = min(i * num_step, total_frame - 1)
        new_lst.append(lst_frame[idx])

    return new_lst

def make_landmark_timestep(hands_results, pose_results):
    c_lm = [0] * (TOTAL_POSE_LANDMARKS * 3 + TOTAL_HAND_LANDMARKS * 3 * TOTAL_HANDS) 
    if pose_results.pose_landmarks:
        for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
            c_lm[idx * 3:(idx + 1) * 3] = [lm.x, lm.y, lm.z]
    
    if hands_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            for idx, lm in enumerate(hand_landmarks.landmark):
                base_idx = TOTAL_POSE_LANDMARKS * 3 + (hand_idx * TOTAL_HAND_LANDMARKS * 3) + (idx * 3)
                c_lm[base_idx:base_idx + 3] = [lm.x, lm.y, lm.z]
    c_lm = transform_to_nose_coordinate(c_lm, NOSE_POSITION)
    return c_lm

def transform_to_nose_coordinate(c_lm, nose_index=0):
    x, y, z = c_lm[nose_index * 3], c_lm[nose_index * 3 + 1], c_lm[nose_index * 3 + 2]
    
    for i in range(0, len(c_lm), 3):
        c_lm[i], c_lm[i + 1], c_lm[i + 2] = c_lm[i] - x, c_lm[i + 1] - y, c_lm[i + 2] - z
    
    return c_lm

def read_vieo(video_path, label):
    cap = cv2.VideoCapture(video_path)
 
    result = []
    lm_list = []
    no_of_frames = 600


    while True:

        ret, frame = cap.read()
        
        if ret:
            # Nhận diện pose
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_results = hands.process(frameRGB)
            pose_results = pose.process(frameRGB)

            if pose_results.pose_landmarks and hands_results.multi_hand_landmarks:
                # Ghi nhận thông số khung xương
                lm = make_landmark_timestep(hands_results, pose_results)
                lm_list.append(lm)
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(frame, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)  
            cv2.imshow("image", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    lm_list = fixed_num_frame(lm_list, NUM_FRAME_PROCESS)
    print('count frame:::', len(lm_list))
    lm_list.append(label)

    return lm_list

def is_file_created_today(file_path):
    # Lấy thời gian hiện tại
    now = datetime.now()
    
    # Lấy thời gian sửa đổi lần cuối của tệp tin
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    print(file_time)
    
    # Kiểm tra xem tệp tin có được tạo vào hôm nay không
    return (now.year, now.month, now.day) == (file_time.year, file_time.month, file_time.day)


mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
pose = mpPose.Pose()
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

video_dataset_path = "Video_dataset"
lst_data = []

for idx_label, subdir in enumerate(os.listdir(video_dataset_path)):
    subdir_path = os.path.join(video_dataset_path, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path) and file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                if is_file_created_today(file_path) and idx_label == 2:
                    data = read_vieo(file_path, idx_label)
                    lst_data.append(data)
                    # with open('dataset.txt', 'w') as f:
                    #     f.write(str(lst_data))
                    # exit()
                

with open(OUTPUT_FILE, 'a') as f:
    for data in lst_data:
            f.write(','.join(map(str, data)) + '\n')

