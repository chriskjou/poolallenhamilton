import cv2

def closest_pocket(ball):
	d = 2000
	for pocket in [(1,1),(2,2)]:
		d_p = np.sqrt((ball[1]-d[0])**2+(ball[2]-d[1])**2)
		d = d_p if d_p < p else d_p
	return d

def ball_to_input(balls):
	for ball in balls:
		d = closest_pocket(ball)
		# TODO
		pass