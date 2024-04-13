
from typing import Any

import cv2
import time
import math
import json
import random
import numpy as np
import pygame
from threading import Thread

class WebcamStream ():
        def __init__(self,width: int, height: int):
            
            #cv2.CAP_MSMF for newer cameras on windows
            self.vcap      = cv2.VideoCapture(1,cv2.CAP_DSHOW)
            self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            self.vcap.set(cv2.CAP_PROP_FPS, 60)
            if self.vcap.isOpened() is False :            
                exit(0)
                
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :                
                exit(0)
                
            self.stopped = True   
            
            self.t = Thread(target=self.update, args=())
            
             
        def isOpened(self):
            return self.vcap.isOpened()
        
        def start(self):           
            self.stopped = False
            self.t.start()
            
        def update(self):
            while True :
                if self.stopped is True :
                    break
                self.grabbed , self.frame = self.vcap.read()
                if self.grabbed is False :                   
                    self.stopped = True
                    break 
            self.vcap.release()
            
        def read(self):
            return self.grabbed ,self.frame  
        
        def stop(self):
            self.stopped = True
            
class Baloon(pygame.sprite.Sprite):

  def __init__(self, startpos, radius, velocity, startdir):
      super().__init__()
      self.pos = pygame.math.Vector2(startpos)
      self.velocity = velocity
      self.dir = pygame.math.Vector2(startdir).normalize()
      self.radius = radius
      self.image = pygame.image.load('pngs/balloonv2.png').convert_alpha()      

  def reflect(self, NV):
      #rect not bouncing
      self.dir = self.dir.reflect(pygame.math.Vector2(NV))
     
  def update(self):
      gravity = 0.05
      self.dir.y += gravity      
      if self.dir.length() > self.velocity:
            self.dir.scale_to_length(self.velocity)
      self.pos += self.dir * self.velocity
      
class Player(pygame.sprite.Sprite):

  def __init__(self, screen, startpos, angle,dis, Left,Right):
      super().__init__()
      self.pos = pygame.math.Vector2(startpos)      
      self.angle_degrees = angle      
      self.min_dis = dis      
      self.wristLeft = Left      
      self.wristRIght = Right     
      self.image = pygame.image.load('pngs/platformv2.png').convert_alpha()      
      self.canvas = screen      
       
  def update(self,posLeftWrist = (0,0), posRightWrist = (0,0)):     
      
      point1 = pygame.math.Vector2(posLeftWrist)
      point2 = pygame.math.Vector2(posRightWrist)
      
      distance = point1.distance_to(point2)
      
      angle = math.atan2(point2.y - point1.y, point2.x - point1.x)  # in radians             
      center = (point1 + point2) / 2
           
      self.min_dis = min(400, max(25,distance))
      self.angle_degrees = math.degrees(angle)
      self.pos = center
      self.wristLeft = point1
      self.wristRIght = point2
  def drawPlayer(self):      
      
      stretched_sprite = pygame.transform.scale(self.image, (int(self.min_dis), 40))
      rotated_sprite = pygame.transform.rotate(stretched_sprite, -self.angle_degrees)
      blit_position = rotated_sprite.get_rect(center=self.pos)
      
      self.canvas.blit(rotated_sprite, blit_position)

class Colliders(pygame.sprite.Sprite):

  def __init__(self, startpos,rect, rotate = False, screen=Any ):
      super().__init__()
      self.pos = pygame.math.Vector2(startpos)     
      self.image = pygame.image.load('pngs/basketv1.png').convert_alpha()     
      self.flip = rotate    
      self.canvas = screen
      self.rect = rect
      
  def draw(self):
     image = pygame.transform.scale(self.image,(300,300))
     imageScaled = pygame.transform.flip(image, self.flip, 0)
     blit_position = image.get_rect(center=self.pos)
     self.rect = pygame.Rect(self.pos.x,self.pos.y,200,200)#blit_position
     
     self.canvas.blit(imageScaled, blit_position)
    

class Questions():
    def __init__(self, startpos, font, color, text, width, screen):
      super().__init__()        
      self.font = font
      self.color = color
      self.text = text
      self.width = width
      self.positions = [startpos, startpos -  pygame.math.Vector2(450,-50), startpos +  pygame.math.Vector2(450, 50)]
      self.canvas = screen
    
    def update(self, text):
        self.text = text
    
    def wrap_text_q(self, text):
        words = text.split(" ")
        lines = []
        current_text = ""
        for word in words:
            tmp_text = current_text + " " + word if current_text else word
            if pygame.font.Font.size(self.font,tmp_text)[0] <= self.width:
                current_text = tmp_text
            else:
                lines.append(current_text)
                current_text = word   
                
        lines.append(current_text)
        return lines        
                
    def draw(self):
       
       for i,text in enumerate(self.text):
           lines = self.wrap_text_q(text)
           pos = self.positions[i].copy()
           for wrapped in lines:
               txt = self.font.render(wrapped, True, self.color[i])
               question_position = txt.get_rect(center=pos)
               self.canvas.blit(txt, question_position)
               pos.y += pygame.font.Font.size(self.font,wrapped)[1]
                           
 
def load_questions(JsonData):    
    GeneratedQuestion = JsonData['Questions'][random.randrange(0, len(JsonData['Questions']))]    
    return [GeneratedQuestion['Question'], GeneratedQuestion['Answer1'], GeneratedQuestion['Answer2']]
 
def imagePyGame(screen, frame, text,font):
   
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    frame = pygame.surfarray.make_surface(frame.swapaxes(1,0))
    #frame = pygame.transform.scale(frame, screen.get_size())
    
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(frame, (0,0))
    screen.blit(text_surface, (10, 10))   
def point_circle(point, circle_center, circle_radius):
    distX = point.x - circle_center.x;
    distY = point.y - circle_center.y;
    distance = np.sqrt( (distX*distX) + (distY*distY) );
    
    return distance <= circle_radius
def line_point(Point1,Point2, ppointx, ppointy):
    
    ppoint = pygame.math.Vector2(ppointx,ppointy)

    d1 = ppoint.distance_to(Point1)
    d2 = ppoint.distance_to(Point2)
    
    lineLen = Point1.distance_to(Point2)
    buffer = 8.;   
    
    if ((d1+d2 >= lineLen - buffer) and (d1+d2 <= lineLen+buffer)) :
       return True    
    
    return False
def circle_line_col(circle_center,circle_radius,customRect = Any,rect = Any, size = Any):
    

    center = rect#pygame.math.Vector2(rect.center)
    if pygame.math.Vector2.length(customRect[0]) <= 0: 
        return False, pygame.math.Vector2(0,1)
    leftEdge  = center + (size * .5) * pygame.math.Vector2.normalize(customRect[0]  - center)
    rightEdge = center + (size * .5) * pygame.math.Vector2.normalize(customRect[1]  - center)
    
    collision_normal = pygame.math.Vector2(rightEdge.y - leftEdge.y,-(rightEdge.x - leftEdge.x))
    collision_normal =  pygame.math.Vector2.normalize(collision_normal)
    
    if (point_circle(leftEdge,circle_center,circle_radius) | point_circle(rightEdge,circle_center,circle_radius)):
        return True, collision_normal

    lineX = rightEdge.x - leftEdge.x
    lineY = rightEdge.y - leftEdge.y
    leng = np.sqrt( (lineX*lineX) + (lineY*lineY) );
    
    dot = ( ((circle_center.x - rightEdge.x)*(leftEdge.x - rightEdge.x)) + ((circle_center.y - rightEdge.y)*(leftEdge.y-rightEdge.y)) ) / pow(leng,2);
    
    closestX = rightEdge.x + (dot * (leftEdge.x - rightEdge.x));
    closestY = rightEdge.y + (dot * (leftEdge.y - rightEdge.y));

    hit = line_point(rightEdge,leftEdge, closestX, closestY)
    
    if hit is False:
        return False, collision_normal;

    distX = closestX - circle_center.x;
    distY = closestY - circle_center.y;
    distance = np.sqrt((distX*distX) + (distY*distY));  
    
    return [distance <= circle_radius, collision_normal]
def circle_box(circle_center, circle_radius, rect = Any):
    closest_x = max(rect.left, min(circle_center[0], rect.right))
    closest_y = max(rect.top, min(circle_center[1], rect.bottom))
    
    distance_squared = (circle_center[0] - closest_x) ** 2 + (circle_center[1] - closest_y) ** 2
    return distance_squared < circle_radius ** 2       