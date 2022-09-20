import keras
import numpy as np
import cv2
model3 =keras.models.load_model('model3.h5')
path=r'E:\3100\css\nav\nav2\static\image\8863_idx5_x51_y1251_class0.jpg'

img_array = cv2.imread(path) 
filee=img_array
filee = np.reshape(filee, (-1, filee.shape[0], filee.shape[1],filee.shape[2]))
filee = filee/255.0
ans=model3.predict(filee)
output = np.argmax(ans[0])
if output==0:
  res='NO BREAST CANCER'
else:
  res='BREAST CANCER'

print(res)