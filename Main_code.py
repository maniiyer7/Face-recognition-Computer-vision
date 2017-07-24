import numpy as np
import cv2,os
from PIL import Image
face_cascade = cv2.CascadeClassifier('/Users/manikandant/Desktop/haarcascade_frontalface_default.xml')
%autosave 25



# Capture series of Images to make databases

cam = cv2.VideoCapture(0)
Id=raw_input('enter your id')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("/Users/manikandant/Desktop/images/."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>50:
        break
cam.release()
cv2.destroyAllWindows()


# Drawing Rectangle box around image faces.

img = cv2.imread('C:/Users/manikandant/Desktop/images/.1.42.jpg',0)
faces = face_cascade.detectMultiScale(img, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,54,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Detecting Face through webcam and drwaing rectangle aroung face.

cap=cv2.VideoCapture(0)
ret,img=cap.read()
#cv2.imshow('windowname',img)
#cv2.waitKey(0)
faces = face_cascade.detectMultiScale(img, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('frame',img)
cv2.waitKey(0)

# Training the Image databases

recognizer = cv2.face.createLBPHFaceRecognizer()
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=face_cascade.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('/Users/manikandant/Desktop/images/')
recognizer.train(faces, np.array(Ids))
recognizer.save('/Users/manikandant/Desktop/images/trainner.yml')

# Loading the recognizer and trying to detect the face and its name

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('/Users/manikandant/Desktop/images/trainner.yml')

cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 0, 0)

#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(im,"Face",(x,y-10),font,0.55,(0,255,0),1)


while(cam.isOpened()):
    ret, im = cam.read()
    if ret:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            
            result = cv2.face.MinDistancePredictCollector()
            recognizer.predict(gray[y: y + h, x: x + w],result, 0)
            Id = result.getLabel()
            conf = result.getDist()
            
            
            #Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<90):
                if(Id==1):
                    Id="Mani"
                elif(Id==2):
                    Id="Venky"
            else:
                Id="Unknown"
            cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor) 
            #cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
        cv2.imshow('im',im) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
cam.release()
cv2.destroyAllWindows()
