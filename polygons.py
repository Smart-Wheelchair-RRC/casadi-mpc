import cv2
import numpy as np
import time

area_threshold = 3

start_time = time.time()

image = cv2.imread('/home/container_user/wheelchair2/src/wheelchair2_navigation/maps/rrc/rrc_lab.pgm')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 150, 255)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Total contours found:", len(contours))

convex_polygons = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > area_threshold:
        hull = cv2.convexHull(contour)
        if len(hull) >= 2 and len(hull) <= 11:
            convex_polygons.append(hull)

print("Convex polygons:", len(convex_polygons))

output_img = image.copy()
cv2.drawContours(output_img, convex_polygons, -1, (0, 255, 0), 2)

end_time = time.time()
print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")

cv2.imshow('Convex Polygons', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()