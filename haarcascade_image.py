import cv2

#roma tomato
roma_cascade = cv2.CascadeClassifier('C://Users//tusha//Desktop//Tushar School Documents//Masters Project//image_classifier//classifier//cascade.xml')

#russet potato
russet_potato_cascade = cv2.CascadeClassifier('C://Users//tusha//Desktop//Tushar School Documents//Masters Project//image_classifier//red_potato_cascade.xml')

#my image file
img = cv2.imread('C:/Users/tusha/Desktop/Tushar School Documents/Masters Project/image_classifier/p/images.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

russet = russet_potato_cascade.detectMultiScale(gray, 1.3, 5)
roma = roma_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in russet:
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Russet potato',(x-w,y-h), font1, 0.5, (255,0,0), 2, cv2.LINE_AA)

for (x,y,w,h) in roma:
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'roma tomato',(x-w,y-h), font1, 0.5, (11,255,255), 2, cv2.LINE_AA)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()