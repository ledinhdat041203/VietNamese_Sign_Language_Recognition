import cv2
import mediapipe as mp
import pandas as pd
from imutils.video import VideoStream

cap = cv2.VideoCapture('video_test/W03990.mp4')
# cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands


pose = mpPose.Pose()
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = hands.process(frameRGB)
        pose_results = pose.process(frameRGB)

        if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
            # # Vẽ khung xương lên ảnh
            # # frame = draw_landmark_on_image(mpDraw, results, frame)
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(frame, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)  
            # Đếm tổng số điểm nhận diện được
            total_points = 0
            if pose_results.pose_landmarks:
                total_points += len(pose_results.pose_landmarks.landmark)
            if hands_results.multi_hand_landmarks:
                print('anh1')
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    total_points += len(hand_landmarks.landmark)
                    print("Total points hand:: ", len(hand_landmarks.landmark))


            # print("Total points: ", total_points)
            print("Total points pose:: ", len(pose_results.pose_landmarks.landmark))
    
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()