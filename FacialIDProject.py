import cv2
import os
import numpy as np
import imutils
import serial
import time
 #CapturaRostro
personName = 'Carlos'
dataPath = '/Users/Jorge./Documents/Reconocimiento Facial/Data'#Ruta donde esté Data
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
 print('Carpeta creada: ',personPath)
 os.makedirs(personPath)
 cap = cv2.VideoCapture(0)
 faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0
while True:
 ret, frame = cap.read()
 if ret == False: break
 frame =  imutils.resize(frame, width=640)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 auxFrame = frame.copy()
 faces = faceClassif.detectMultiScale(gray,1.3,5)
 for (x,y,w,h) in faces: cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
 rostro = auxFrame[y:y+h,x:x+w]
 rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
 cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
 count = count + 1
 
 cv2.imshow('frame',frame)
 k =  cv2.waitKey(1)
 if k == 27 or count >= 300: break
cap.release()
cv2.destroyAllWindows()
 #EntrenarReconocimientoFacial
dataPath = '/Users/Jorge./Documents/Reconocimiento Facial/Data'#Ruta de Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
if '.DS_Store' in peopleList:
 peopleList.remove('.DS_Store')
 labels = []
 facesData = []
 label = 0
 for nameDir in peopleList:
  personPath = dataPath + '/' + nameDir
  print('Leyendo imágenes')
 for fileName in os.listdir(personPath):
  print('Rostros: ', nameDir + '/' + fileName)
  labels.append(label)
  facesData.append(cv2.imread(personPath+'/'+fileName,0))
  image = cv2.imread(personPath+'/'+fileName,0)
 cv2.imshow('image',image)
 cv2.waitKey(10)
 label = label + 1
 #print('labels= ',labels)
 #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
#print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
 face_recognizer = cv2.face.FisherFaceRecognizer_create()
 
 print("Entrenando reconocimiento...")
 face_recognizer.train(facesData, np.array(labels))
 # Almacenando el modelo
#EnviarDatosSerial
#ReconocimientoFacial
 face_recognizer.write('modeloFisherFace.xml')
 print("Modelo almacenado...")
ser = serial.Serial('/dev/cu.usbmodem14101', 9600, timeout=1)
time.sleep(2)
dataPath = '/Users/Jorge./Documents/Reconocimiento Facial/Data' #Ruta de Data
imagePaths = os.listdir(dataPath)
if '.DS_Store' in imagePaths:
 imagePaths.remove('.DS_Store')
 print('imagePaths=',imagePaths)
face_recognizer = cv2.face.FisherFaceRecognizer_create()
# Leyendo modelo xml')
face_recognizer.read('modeloFisherFace.xml')
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.')
while True:
 ret,frame = cap.read()
 if ret == False: break
 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
auxFrame = gray.copy()
faces = faceClassif.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
 rostro = auxFrame[y:y+h,x:x+w]
 rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
 result = face_recognizer.predict(rostro)
 cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,
 (255,255,0),1,cv2.LINE_AA)
 if result[1] < 500:
  cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
  cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
  if '{}'.format(imagePaths[result[0]]) =='Jorge':
    ser.write(b'J')
  if '{}'.format(imagePaths[result[0]]) =='Carlos':
    ser.write(b'C')
  else:(0,0,255),(1,cv2.LINE_AA)
  cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,)
  cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
  ser.write(b'D')
  cv2.imshow('frame',frame)
  k = cv2.waitKey(1)
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()
ser.close()
 