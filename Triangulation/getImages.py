import cv2
cap = cv2.VideoCapture(0)
num = 0
while cap.isOpened():
    succes, img = cap.read()
    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('Triangulation/images' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)


cap.release()
cv2.destroyAllWindows()