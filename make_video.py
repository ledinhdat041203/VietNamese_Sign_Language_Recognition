import cv2
import time
import os
from imutils.video import VideoStream
from datetime import datetime

# Thông số
now = datetime.now()
label = "example_sign_" + now.strftime("%Y%m%d_%H%M%S")  # Định dạng timestamp cho tên file
duration = 2  # Thời gian quay video (giây)
output_folder = "Video_dataset"
work = "Xin_Loi"
print(label)

# Kiểm tra thư mục lưu trữ
os.makedirs(f"{output_folder}/{work}", exist_ok=True)

# Khởi tạo VideoStream
cap = VideoStream(src=1).start()
time.sleep(2.0)  # Đợi camera khởi động
start_time = time.time()
# Lấy thông số từ camera
frame = cap.read()
h, w = frame.shape[:2]
fps = 60  # Điều chỉnh FPS nếu cần

# Khởi tạo VideoWriter để lưu video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(f"{output_folder}/{work}", f'{label}.avi'), fourcc, fps, (w, h))


while (time.time() - start_time) < duration:
    frame = cap.read()
    if frame is None:
        break

    out.write(frame)
    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
out.release()
cv2.destroyAllWindows()
