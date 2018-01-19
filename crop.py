
### SLIDER SETUP
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

crop_img = image[60:850,40:1620]