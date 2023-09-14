import pygame,sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

pygame.init()

# Define some colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)

# Define labels
MODEL=load_model('model.h5') # Load the model
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Define window size
WINDOWSIZx = 640
WINDOWSIZy = 480

IMAGESAVE=False
PREDICT=True

BOUNDAING=5
DISPLAYSURFACE= pygame.display.set_mode((640,480))
WHITE_INT = DISPLAYSURFACE.map_rgb(WHITE)

# set the pygame window Caption
pygame.display.set_caption('Handwritten Digit Recognition')

isWriting = False

numbr_xcord=[]
numbr_ycord=[]
img_cnt=1

while True:

    for event in  pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if event.type==MOUSEMOTION and isWriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURFACE,WHITE,(xcord,ycord),4,0)
            numbr_xcord.append(xcord)
            numbr_ycord.append(ycord)
            
        if event.type==MOUSEBUTTONDOWN:
            isWriting=True

        if event.type==MOUSEBUTTONUP:
            isWriting=False
            numbr_xcord=sorted(numbr_xcord)
            numbr_ycord=sorted(numbr_ycord)

            rect_min_x,rect_max_x=max(numbr_xcord[0]-BOUNDAING,0),min(WINDOWSIZx,numbr_xcord[-1]+BOUNDAING)
            rect_min_y,rect_max_y=max(numbr_ycord[0]-BOUNDAING,0),min(WINDOWSIZy,numbr_ycord[-1]+BOUNDAING)

            numbr_xcord=[]
            numbr_ycord=[]
            img_arr=np.array(pygame.PixelArray(DISPLAYSURFACE))

            if IMAGESAVE:
                # cv2.imwrite('image.png')
                cv2.imwrite('image.png', img_arr)
                img_cnt+=1
            
            if PREDICT:
                # image=cv2.resize(img_arr,(28,28))
                image = cv2.resize(np.uint8(img_arr), (28, 28))
                image=np.pad(image,(10,10),'constant',constant_values=0)
                image=cv2.resize(image,(28,28))/WHITE_INT
                # labe=str(np.argmax(MODEL.predict(image.reshape(1,28,28,1))) )
                # labe=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))]).title()
                labe=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))]).title()
                pygame.draw.rect(DISPLAYSURFACE,RED,(rect_min_x,rect_min_y,rect_max_x-rect_min_y,rect_max_y-rect_min_y),3)

            if event.type==KEYDOWN:
                if event.unicode=='N':
                    DISPLAYSURFACE.fill(BLACK)
        
        pygame.display.update()


