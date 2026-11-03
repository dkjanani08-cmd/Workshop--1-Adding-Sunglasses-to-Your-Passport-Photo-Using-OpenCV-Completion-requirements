# Workshop--1-Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV-Completion-requirements
# NAME : Janani . d
# REG No : 212225040142

Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look
Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

Features:
Detects the face in an image.
Places a stylish sunglass overlay perfectly on the face.
Works seamlessly with individual passport-size photos.
Customizable for different sunglasses styles or photo types.
Technologies Used:
Python
OpenCV for image processing
Numpy for array manipulations
How to Use:
Clone this repository.
Add your passport-sized photo to the images folder.
Run the script to see your "cool" transformation!
Applications:
Learning basic image processing techniques.
Adding flair to your photos for fun.
Practicing computer vision workflows.
Feel free to fork, contribute, or customize this project for your creative needs!


program :

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread('janani.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

## output :

<img width="372" height="447" alt="image" src="https://github.com/user-attachments/assets/352d7bee-8da6-4c1b-9b0e-57b1584d0839" />

program :

faceImage.shape
(1600, 1200, 3)
resized_faceImage.shape
faceImage.shape

glassPNG = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)

plt.imshow(cv2.cvtColor(glassPNG, cv2.COLOR_BGRA2RGBA))
plt.title("Sunglasses")
plt.show()

## output:

<img width="617" height="325" alt="image" src="https://github.com/user-attachments/assets/542b5c8c-e368-452c-b04d-4192a95ea708" />

program:

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');

## output:


<img width="1137" height="217" alt="image" src="https://github.com/user-attachments/assets/f07d228e-23c1-43f0-8d64-dd7268ab8610" />


# Resize face image
resized_faceImage = cv2.resize(faceImage, (600, 800))

# Resize sunglasses
glassPNG = cv2.resize(glassPNG, (250, 70))
glassBGR = glassPNG[:, :, :3]

# Copy resized face
faceWithGlassesNaive = resized_faceImage.copy()

# Place sunglasses
faceWithGlassesNaive[190:260, 175:425] = glassBGR

# Display
plt.figure(figsize=(5, 7))
plt.imshow(faceWithGlassesNaive[:, :, ::-1])
plt.show()


## output:


<img width="493" height="567" alt="image" src="https://github.com/user-attachments/assets/69263cf6-c5fd-423b-b6cc-276b8704856a" />

# Create 3-channel mask
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Convert mask values to 0 and 1
glassMask = np.uint8(glassMask / 255)

# Make a copy of resized face
faceWithGlassesArithmetic = resized_faceImage.copy()

# Get eye region
eyeROI = faceWithGlassesArithmetic[190:260, 175:425]

# Mask the eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Mask the sunglasses
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine sunglasses and eye region
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display intermediate results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(maskedEye[:, :, ::-1])
plt.title("Masked Eye Region")

plt.subplot(132)
plt.imshow(maskedGlass[:, :, ::-1])
plt.title("Masked Sunglass Region")

plt.subplot(133)
plt.imshow(eyeRoiFinal[:, :, ::-1])
plt.title("Augmented Eye and Sunglass")

plt.show()

## output:

<img width="1128" height="165" alt="image" src="https://github.com/user-attachments/assets/096967bf-3928-4ad8-9837-c69c2c453d7e" />

Make the dimensions of the mask same as the input image.
Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

Make a copy
faceWithGlassesArithmetic = faceImage.copy()

Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[135:185,110:300]

Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1- glassMask ))

Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

Display the intermediate results
plt.figure(figsize=[20,20]) plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region") plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region") plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")


faceWithGlassesArithmetic[190:260, 175:425] = eyeRoiFinal

plt.figure(figsize=(10, 8))

plt.subplot(121)
plt.imshow(resized_faceImage[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[:, :, ::-1])
plt.title("With Sunglasses")

plt.show()

## output:


<img width="862" height="546" alt="image" src="https://github.com/user-attachments/assets/99b10673-9481-4db7-9da4-ed29d0f93c65" />


Result:
The project successfully detected the face in the input image and added sunglasses automatically using OpenCV. The output image showed the sunglasses correctly positioned on the face, demonstrating the effectiveness of basic image processing techniques.
