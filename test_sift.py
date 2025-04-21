import cv2 as cv 
import SIFT 
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv.imread(cv.samples.findFile("graf1.png"))  # Replace with any image path
    if img is None:
        raise FileNotFoundError("Image not found. Provide a valid image path.")

    sift = SIFT(img)

    # Generate base image
    base_img = sift.generate_base_image(sift.image)

    # Compute number of octaves
    num_octaves = sift.compute_number_of_octaves(base_img.shape)

    # Generate Gaussian and DoG pyramids
    gaussian_pyramid = sift.generate_gaussian_images(base_img, num_octaves=num_octaves)
    dog_pyramid = sift.generate_dog_images(gaussian_pyramid)

    # Find keypoints
    keypoints = sift.find_scale_space_extrema(dog_pyramid)

    # Draw keypoints on the image
    output_img = img.copy()
    for octave, scale, x, y in keypoints:
        cv.circle(output_img, (int(x * (2 ** octave)), int(y * (2 ** octave))), 1, (0, 255, 0), -1)

    # Show result using matplotlib
    plt.imshow(cv.cvtColor(output_img, cv.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints")
    plt.axis("off")
    plt.show()
