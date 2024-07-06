import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
import json

TOTAL_POSE_LANDMARKS = 33 
TOTAL_HAND_LANDMARKS = 21  
TOTAL_HANDS = 2   
NUM_FRAME_PROCESS = 25
TOTAL_COORDINATES  = TOTAL_POSE_LANDMARKS * 3 + TOTAL_HAND_LANDMARKS * 3 * TOTAL_HANDS
NOSE_POSITION = 0

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
def draw_class_on_image(label, img, x ,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, lm_list, lst_label):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    result = np.argmax(results)
    label = lst_label[str(result)]
    print(":",lst_label[str(result)])
    return 

def fixed_num_frame(lst_frame, num_frame):
    total_frame = len(lst_frame)
    
    num_step = max(total_frame // (num_frame - 1), 1)

    new_lst = []

    for i in range(num_frame):
        idx = min(i * num_step, total_frame - 1)
        new_lst.append(lst_frame[idx])

    return new_lst

model = tf.keras.models.load_model("model.h5")
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
pose = mpPose.Pose()
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
label = "Warmup...."

cap = cv2.VideoCapture(0)
with open("label.json", "r") as file:
    lst_label = json.load(file)
# cap = cv2.VideoCapture('Video_dataset/Cam_on/example_sign_20240704_122748.avi')
lm_list = []
start = time.time()
while True:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands.process(frameRGB)
        pose_results = pose.process(frameRGB)
        end = time.time()
        if pose_results.pose_landmarks and hands_results.multi_hand_landmarks:
            lm = make_landmark_timestep(hands_results, pose_results)
            if len(lm) == TOTAL_COORDINATES :
                lm_list.append(lm)
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(frame, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)  
            # if len(lm_list) == NUM_FRAME_PROCESS:
            #     print("Start detect....")
            #     t1 = threading.Thread(target=detect, args=(model, lm_list,))
            #     t1.start()
            #     lm_list = []
        if end - start  >= 4:
            start = time.time()
            if len(lm_list)>=NUM_FRAME_PROCESS:
                lm_list = fixed_num_frame(lm_list, NUM_FRAME_PROCESS)
                print("Start detect....")
                t1 = threading.Thread(target=detect, args=(model, lm_list, lst_label))
                t1.start()
            lm_list.clear()
        img = draw_class_on_image(label, frame, 10, 30)
        img = draw_class_on_image(f'total_frame: {len(lm_list)}', frame, 300, 30)
        img = draw_class_on_image(f'GIAY: {end - start}', frame, 300, 300)
        

        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('x'):
            lm_list = []
    else:
        break
cap.release()
cv2.destroyAllWindows()
