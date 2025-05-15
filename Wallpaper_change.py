import ctypes
import time
import cv2
import os
from PIL import Image 
import numpy as np 


def take_user_face_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5
         
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    id = 2
    img_id = 0
     
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
            file_name_path = os.path.join("C:\python projects\data", f"user.{id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
             
            cv2.imshow("Cropped face", face)
             
        if cv2.waitKey(1)==13 or int(img_id)==200: #13 is the ASCII character of Enter
            break
             
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")



 
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")


 
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    if len(features) == 0:
        ctypes.windll.user32.SystemParametersInfoW(20, 0, "C:\\python projects\\unauthorised.jpg", 3)

    for (x,y,w,h) in features:
        # cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
         
        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
         
        if confidence>50:
            if id==1 or id==2:
                # ctypes.windll.user32.SystemParametersInfoW(20, 0, "C:\\python projects\\unauthorised.jpg", 3)
                ctypes.windll.user32.SystemParametersInfoW(20, 0, "C:\\python projects\\My_wall.jpg", 3)

        else:
            # ctypes.windll.user32.SystemParametersInfoW(20, 0, "C:\\python projects\\My_wall.jpg", 3)
            ctypes.windll.user32.SystemParametersInfoW(20, 0, "C:\\python projects\\unauthorised.jpg", 3)
    return img
 


if __name__ == "__main__":
    # take_user_face_dataset() 
    # train_classifier("data")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, img = video_capture.read()
        img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        # cv2.imshow("face Detection", img)
        
        if cv2.waitKey(1) and 0xFF == ord('q'):
            print("Exiting video capture.")
            break
    video_capture.release()
    cv2.destroyAllWindows()
    time.sleep(5)
