import csv
import cv2
import numpy as np
import os
import math
from random import shuffle

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

recover_samples = []
with open('data_recover/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		recover_samples.append(line)

shuffle(samples)
shuffle(recover_samples)

image_paths = []
angles = []
for sample in samples:
    name = 'data/IMG/'+ sample[0].split('/')[-1]
    image_paths.append(name)
    center_angle = float(sample[3])
    angles.append(center_angle)

for sample in recover_samples:
    name = 'data_recover/IMG/'+ sample[0].split('/')[-1]
    image_paths.append(name)
    center_angle = float(sample[3])
    angles.append(center_angle)

print("size of angles:", len(angles))
print("size of image_paths", len(image_paths))
print("angles[14]", angles[14])
print("image_paths[14]", image_paths[14])
image = cv2.imread(image_paths[14])
cv2.imshow("im1", image)
cv2.waitKey(0)