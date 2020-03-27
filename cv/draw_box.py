# import the necessary packages
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-tl", "--top-left", nargs='+', type=int, required=True,
        help="coords to top left corner of box")
ap.add_argument("-br", "--bottom-right", nargs='+', type=int, required=True,
        help="coords to bottom right corner of box")
args = vars(ap.parse_args())

# parse the coords into a tuple
tl_coords = tuple(args["top_left"])
br_coords = tuple(args["bottom_right"])

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

color = ( 255, 0, 0 )
cv2.rectangle(image, tl_coords, br_coords, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

