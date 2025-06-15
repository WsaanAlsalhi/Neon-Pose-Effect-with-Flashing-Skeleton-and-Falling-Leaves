import cv2
import numpy as np
from ultralytics import YOLO
import random
import time

#########################################
# Color Palette for Skeleton (Updated)
#########################################
skeleton_colors = [
    (30, 122, 98), 
    (108, 156, 32), 
    (110, 190, 260),  
]

def gradient_color(start_color, end_color, ratio):
    return (
        int(start_color[0] + (end_color[0] - start_color[0]) * ratio),
        int(start_color[1] + (end_color[1] - start_color[1]) * ratio),
        int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
    )

def get_leaf_color(y, height):
    ratio = y / height
    return gradient_color((32, 108, 90), (255, 255, 0), ratio)

def get_skeleton_color(distance, flash_intensity):
    base_color = random.choice(skeleton_colors)
    r, g, b = base_color
    flash_effect = random.uniform(1 - flash_intensity, 1 + flash_intensity)
    r = int(min(255, max(0, r * flash_effect)))
    g = int(min(255, max(0, g * flash_effect)))
    b = int(min(255, max(0, b * flash_effect)))
    return (r, g, b)

#########################################
# Leaf Class
#########################################
class Leaf:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = random.randint(0, width)
        self.y = random.randint(-height, 0)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(3, 7)
        self.size = random.randint(15, 30)  # Medium size between 15 and 30 pixels
        self.color = (32, 108, 90)
        self.angle = random.uniform(0, 360)
        self.angular_velocity = random.uniform(-5, 5)

    def move(self, skeleton_points):
        force_x, force_y = 0, 0

        for point in skeleton_points:
            x, y = point
            distance = np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

            if distance < self.size * 6:
                angle_to_skeleton = np.arctan2(self.y - y, self.x - x)
                strength = max(0, (self.size * 6 - distance) / (self.size * 6))
                force_x += np.cos(angle_to_skeleton) * strength * 15
                force_y += np.sin(angle_to_skeleton) * strength * 15

        self.vx += force_x
        self.vy += force_y

        self.x += self.vx
        self.y += self.vy
        self.angle = (self.angle + self.angular_velocity) % 360

        # Reset leaf to top if it falls off screen
        if self.y > self.height:
            self.y = random.randint(-50, -10)
            self.x = random.randint(0, self.width)

        # Prevent leaves from going out of horizontal bounds
        if self.x < 0:
            self.x = 0
            self.vx *= -0.9
        elif self.x > self.width:
            self.x = self.width
            self.vx *= -0.9

    def draw(self, image):
        center = (int(self.x), int(self.y))
        axes = (self.size // 2, self.size // 4)
        self.color = get_leaf_color(self.y, self.height)
        cv2.ellipse(image, center, axes, self.angle, 0, 360, self.color, -1)

#########################################
# Skeleton definition
#########################################
skeleton = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12)
]

def get_xy(point):
    flat = np.ravel(point)
    return int(flat[0]), int(flat[1])

#########################################
# YOLO Pose setup
#########################################
model = YOLO('yolov8m-pose.pt')
cap = cv2.VideoCapture(0)

# Set camera resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam!")
    exit()

# Initialize leaves
ret, test_frame = cap.read()
if not ret:
    print("❌ Error: Cannot capture initial frame!")
    cap.release()
    exit()
test_frame = cv2.flip(test_frame, 1)
test_h, test_w, _ = test_frame.shape

num_leaves = 50
leaves = [Leaf(test_w, test_h) for _ in range(num_leaves)]

# Create a fullscreen window
cv2.namedWindow("Neon Pose Effect with Flashing Skeleton and Falling Leaves", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Neon Pose Effect with Flashing Skeleton and Falling Leaves", cv2.WND_PROP_FULLSCREEN, 1)

#########################################
# Main loop
#########################################
flash_intensity = 0.4
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame from camera!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Set black background
    black_bg = np.zeros((h, w, 3), dtype=np.uint8)

    results = model(frame, conf=0.3)

    skeleton_points = []
    for result in results:
        if result.keypoints is None:
            continue
        keypoints = result.keypoints.xy.cpu().numpy()

        for person in keypoints:
            for idx, (pt1, pt2) in enumerate(skeleton):
                if pt1 < len(person) and pt2 < len(person):
                    x1, y1 = get_xy(person[pt1])
                    x2, y2 = get_xy(person[pt2])
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        distance = np.linalg.norm([x2 - x1, y2 - y1])
                        skeleton_color = get_skeleton_color(distance, flash_intensity)
                        cv2.line(black_bg, (x1, y1), (x2, y2), skeleton_color, 4)

            skeleton_points = [(int(x), int(y)) for x, y in person if x > 0 and y > 0]
            for x, y in skeleton_points:
                distance = np.linalg.norm([x, y])
                skeleton_color = get_skeleton_color(distance, flash_intensity)
                cv2.circle(black_bg, (x, y), 8, skeleton_color, -1)

    for leaf in leaves:
        leaf.move(skeleton_points)
        leaf.draw(black_bg)

    cv2.imshow("Neon Pose Effect with Flashing Skeleton and Falling Leaves", black_bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

