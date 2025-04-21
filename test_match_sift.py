import cv2
import numpy as np
import matplotlib.pyplot as plt
from SIFT import SIFT  


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    matched_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    matched_image[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    matched_image[:h2, w1:] = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, (idx1, idx2) in enumerate(matches[:max_matches]):
        x1, y1, _ = kp1[idx1]
        x2, y2, _ = kp2[idx2]

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(matched_image, pt1, pt2, color, 1)
        cv2.circle(matched_image, pt1, 3, color, -1)
        cv2.circle(matched_image, pt2, 3, color, -1)

    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))  # ✅ Fix color

    plt.title('SIFT Keypoint Matches')
    plt.axis('off')
    plt.show()





def get_good_matches(descriptors1, descriptors2, ratio_thresh=0.75):
    descriptors2 = np.array(descriptors2)
    good_matches = []
    for i, d1 in enumerate(descriptors1):
        distances = np.sum((descriptors2 - d1) ** 2, axis=1)
        nearest = np.argsort(distances)
        if distances[nearest[0]] < ratio_thresh * distances[nearest[1]]:
            good_matches.append((i, nearest[0]))
    return good_matches


if __name__ == "__main__":
    
    image1_path = "C:/Computer_Vision/task2_canny&activecountour/Edge-and-boundary-detection-Hough-transform-and-SNAKE-/Images/cat.jpg"
    image2_path = "C:/Computer_Vision/task2_canny&activecountour/Edge-and-boundary-detection-Hough-transform-and-SNAKE-/Images/cat_part_mod2.jpg"

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    sift1 = SIFT(image1)
    raw_kp1 = sift1.find_scale_space_extrema()
    oriented_kp1 = sift1.assign_orientation(raw_kp1)
    desc1 = sift1.compute_descriptors(oriented_kp1)

    sift2 = SIFT(image2)
    raw_kp2 = sift2.find_scale_space_extrema()
    oriented_kp2 = sift2.assign_orientation(raw_kp2)
    desc2 = sift2.compute_descriptors(oriented_kp2)

    matches = get_good_matches(desc1, desc2)
    print(f"Found {len(matches)} good matches")

    draw_matches(image1, oriented_kp1, image2, oriented_kp2, matches)
