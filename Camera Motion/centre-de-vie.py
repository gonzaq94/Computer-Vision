import cv2
import numpy as np
from drawlines import drawlines
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt

# Using cv2.imread() method 
img1 = cv2.imread("data/real-pair-of-images/centre-de-vie1.jpg")
img2 = cv2.imread("data/real-pair-of-images/centre-de-vie2.jpg")

f = 752.253
Cx = 626.696
Cy = 364.502
K1 = 0.0884906
K2 = -0.225692
K3 = 0.0988774
TD1 = 0
TD2 = 0

K = np.array([f, 0, Cx,0, f, Cy, 0,0,1]).reshape(3,3)
distCoeffs = np.array([K1,K2,0,0])

# undistort image
img1_undis = cv2.undistort(img1,K,distCoeffs)
img2_undis = cv2.undistort(img2,K,distCoeffs)

# find keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_undis,None)
kp2, des2 = sift.detectAndCompute(img2_undis,None)

img1_sift = cv2.drawKeypoints(img1_undis,kp1,img1_undis)
img2_sift = cv2.drawKeypoints(img2_undis,kp2,img2_undis)

plt.imshow(img1_sift)
plt.title('Img1 keypoints')
plt.show()
cv2.imwrite('output_images/real-pair-of-images/img1_keypoints.jpg',img1_sift)

plt.imshow(img2_sift)
plt.title('Img2 keypoints')
plt.show()
cv2.imwrite('output_images/real-pair-of-images/img2_keypoints.jpg',img2_sift)


# Match descriptors.
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = np.zeros(img1_undis.shape)
img3 = cv2.drawMatchesKnn(img1_undis,kp1,img2_undis,kp2,good,img3,flags=2)

plt.imshow(img3)
plt.title('Matches')
plt.show()
cv2.imwrite('output_images/real-pair-of-images/matches.jpg',img3)
#cv2.imwrite('img_matched.jpg',img3)

good = []
pts1 = []
pts2 = []

# ratio test as per Ldef drawlines(img1,img2,lines,pts1,pts2):

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# calculate the F matrix

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img1_epip,img6 = drawlines(img1_undis,img2_undis,lines1,pts1,pts2)

plt.imshow(img1_epip)
plt.title('Img1 epipolar lines')
plt.show()
cv2.imwrite('output_images/real-pair-of-images/img1_epipolar.jpg',img1_epip)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2,F)
lines2 = lines2.reshape(-1,3)
img2_epip,img7 = drawlines(img2_undis,img1_undis,lines2,pts2,pts1)

plt.imshow(img2_epip)
plt.title('Img2 epipolar lines')
plt.show()
cv2.imwrite('output_images/real-pair-of-images/img2_epipolar.jpg',img2_epip)

#Essential matrix

pts1 = []
pts2 = []
good = []

# ratio test as per Ldef drawlines(img1,img2,lines,pts1,pts2):

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts2,pts1,cv2.FM_RANSAC)

E = np.matmul(np.matmul(np.transpose(K),F),K)
R = np.zeros((3,3))
t = np.zeros((3,1))
cv2.recoverPose(E, pts2, pts1, K, R, t, mask);

T = np.concatenate((np.transpose(R),-np.matmul(np.transpose(R),t)), axis=1)
r = Rot.from_matrix(R)
T = np.concatenate((T,np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0)
print('Relative motion matrix: ',T)

angles = r.as_euler('xyz', degrees=True)
print('Euler angles: ',angles)
print('Translation direction: ',t)
