import pygame
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
import sys

# Define some colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)

# Define window size
WINDOWSIZx = 640
WINDOWSIZy = 480

MODEL=load_model('../Model/model.h5') # Load the model
LABELS ={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
IMAGESAVE=False
PREDICT=True
BOUNDAING=5


pygame.init()

FONT=pygame.font.SysFont('freesaansbold.tff', 18)
DISPLAYSURFACE=pygame.display.set_mode((WINDOWSIZx, WINDOWSIZy))
# WHILE_INI=DISPLAYSURFACE.map_rgb(WHITE)
pygame.display.set_caption('Handwritten Digit Recognition')

isWriting=False

Number_xcord=[]
Number_ycord=[]
img_cnt=1

while True:

        for event in  pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type==MOUSEMOTION and isWriting:
                xcord,ycord=event.pos
                pygame.draw.circle(DISPLAYSURFACE,WHITE,(xcord,ycord),4,0) # Draw a white circle at mouse position

                Number_xcord.append(xcord)
                Number_ycord.append(ycord)

            if event.type==MOUSEBUTTONDOWN:
                isWriting=True

            if event.type==MOUSEBUTTONUP:
                isWriting=False
                Number_xcord=sorted(Number_xcord)
                Number_ycord=sorted(Number_ycord)

                rect_min_x,rect_max_x=max(Number_xcord[0]-BOUNDAING,0),min(WINDOWSIZx,Number_xcord[-1]+BOUNDAING)
                rect_min_y,rect_max_y=max(Number_ycord[0]-BOUNDAING,0),min(WINDOWSIZy,Number_ycord[-1]+BOUNDAING)

                Number_xcord=[]
                Number_ycord=[]

                img_arr=np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype('float32') # Convert the drawn digit to a numpy array

                if IMAGESAVE:
                    cv2.imwrite("img.png")
                    img_cnt+=1

                if PREDICT:
                    image=cv2.resize(img_arr,(28,28))
                    image=np.pad(image,(10,10),'constant',constant_values=0)
                    image=cv2.resize(image,(28,28))/255

                    label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                    text=FONT.render(label,True,RED,WHITE)
                    textRect=text.get_rect()
                    textRect.left,textRect.bottom=rect_min_x,rect_max_y
                    
                    pygame.draw.rect(DISPLAYSURFACE,RED,(rect_min_x,rect_min_y,rect_max_x-rect_min_x,rect_max_y-rect_min_y),3)
                    DISPLAYSURFACE.blit(text,textRect)

                if event.type==KEYDOWN:
                    if event.unicode=="n":
                        DISPLAYSURFACE.fill(BLACK)
            
            pygame.display.update()

                
                 
            
