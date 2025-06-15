import cv2
import numpy as np
from ultralytics import YOLO
import random

# تحميل نموذج YOLOv8 Pose
model = YOLO('yolov8m-pose.pt')

# فتح كاميرا الويب
cap = cv2.VideoCapture(0)

# إنشاء خلفية سوداء لحفظ التأثير السابق
trail_frame = None

# تحميل صورة نبات (ورقة) لاستخدامها في التأثير
leaf_image = cv2.imread('C:/Users/HP/Pictures/codes/images/leaf.png', cv2.IMREAD_UNCHANGED)
 # تحميل صورة ورقة شفافة
leaf_height, leaf_width = leaf_image.shape[:2]

# إضافة أوراق عشوائية
def add_random_leaf(frame):
    leaf_x = random.randint(0, frame.shape[1] - leaf_width)
    leaf_y = random.randint(0, frame.shape[0] - leaf_height)
    return (leaf_x, leaf_y)

# تأثير النباتات المتسلقة
def add_climbing_plants(frame, keypoints):
    # إضافة أوراق تتسلق الأذرع والساقين
    for keypoint in keypoints:
        x, y = int(keypoint[0][0]), int(keypoint[0][1])
        if 0 < x < frame.shape[1] and 0 < y < frame.shape[0]:
            leaf_x, leaf_y = add_random_leaf(frame)
            leaf_resized = cv2.resize(leaf_image, (leaf_width // 3, leaf_height // 3))  # تصغير الورقة
            # إضافة الورقة على النقطة
            frame[y:y+leaf_resized.shape[0], x:x+leaf_resized.shape[1]] = leaf_resized
    return frame

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # قلب الإطار أفقياً ليكون أكثر واقعية مثل المرآة
    frame = cv2.flip(frame, 1)

    # تشغيل YOLO لاستخراج الهيكل العظمي
    results = model(frame, conf=0.3)

    # إنشاء صورة سوداء بنفس حجم الإطار
    black_background = np.zeros_like(frame)

    # التعامل مع الهيكل العظمي والنقاط العظمية
    for result in results:
        if result.keypoints is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        # إضافة تأثير النباتات المتسلقة على الهيكل العظمي
        frame = add_climbing_plants(frame, keypoints)

        # رسم الهيكل العظمي (النقاط والخطوط)
        for keypoint in keypoints:
            for x, y in keypoint:
                x, y = int(x), int(y)
                if x > 0 and y > 0:
                    color = (np.random.randint(100, 255), np.random.randint(50, 200), 255)
                    cv2.circle(black_background, (x, y), 8, color, -1)

        skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12)
        ]

        for keypoint in keypoints:
            for pt1, pt2 in skeleton:
                if pt1 < len(keypoint) and pt2 < len(keypoint):
                    x1, y1 = int(keypoint[pt1][0]), int(keypoint[pt1][1])
                    x2, y2 = int(keypoint[pt2][0]), int(keypoint[pt2][1])
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(black_background, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # دمج الإطارات لإنشاء تأثير الذيل (Glow Trail)
    if trail_frame is None:
        trail_frame = black_background.copy()
    else:
        cv2.addWeighted(trail_frame, 0.8, black_background, 0.3, 0, trail_frame)

    # دمج الخلفية السوداء مع تأثير الذيل
    output_frame = cv2.addWeighted(black_background, 0.8, trail_frame, 0.2, 0)

    # عرض النتيجة
    cv2.imshow("Neon Pose with Climbing Plants", output_frame)

    # الخروج عند الضغط على "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الكاميرا
cap.release()
cv2.destroyAllWindows()

