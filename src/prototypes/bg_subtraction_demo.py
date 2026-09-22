import cv2
import numpy as np

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # 2. Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # fill holes

    # 3. Find contours (candidate blobs)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # filter tiny noise
            continue

        # 4. Fit a circle (ball check)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = area / (np.pi * radius**2)

        if circularity > 0.6 and radius > 5:  # roughly circular = ball
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

            # 5. Collision zone check
            if is_in_collision_zone(center, radius):
                print(f"COLLISION at {center}")

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", fg_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



def is_in_collision_zone(center, radius, zone_polygon):
    """
    zone_polygon: np.array of shape (N, 2) defining the surface boundary
    """
    result = cv2.pointPolygonTest(zone_polygon, center, measureDist=False)
    return result >= 0  # center is inside zone

# Or simpler rectangular zone:
def is_in_collision_zone(center, radius):
    ZONE = (100, 200, 400, 500)  # x1, y1, x2, y2
    cx, cy = center
    return ZONE[0] < cx < ZONE[2] and ZONE[1] < cy < ZONE[3]