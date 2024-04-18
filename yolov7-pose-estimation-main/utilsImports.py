
from typing import Any

import os
import cv2
import time
import datetime
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
        self.image = pygame.image.load('pngs/Balloon.png').convert_alpha()      
    
    def reflect(self, NV):    
        self.dir = self.dir.reflect(pygame.math.Vector2(NV))
       
    def update(self,segments, event):
        gravity = 0.04
        self.dir.y += gravity      
        if self.dir.length() > self.velocity:
            self.dir.scale_to_length(self.velocity)
        self.pos += self.dir * self.velocity
        for i in range(5):
            collision = collideWithSegments(self.pos, segments)
            if(collision[0] > 0):
                self.dir = (collision[1] * 3.5)    
                pygame.event.post(pygame.event.Event(event, value=collision[2].id))
                break
                  
class Player(pygame.sprite.Sprite):

  def __init__(self, screen, startpos, angle,dis, Left,Right):
      super().__init__()
      self.id = 0
      self.pos = pygame.math.Vector2(startpos)      
      self.angle_degrees = angle      
      self.min_dis = dis      
      self.left_right_points = [pygame.math.Vector2(Left), pygame.math.Vector2(Right)]      
         
      self.image = pygame.image.load('pngs/Paddle.png').convert_alpha()      
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
      self.left_right_points = [point1, point2]
  def get_position(self):
      return  self.left_right_points   
  def drawPlayer(self):      
      
      stretched_sprite = pygame.transform.scale(self.image, (int(self.min_dis), 40))
      rotated_sprite = pygame.transform.rotate(stretched_sprite, -self.angle_degrees)
      blit_position = rotated_sprite.get_rect(center=self.pos)
      
      self.canvas.blit(rotated_sprite, blit_position)
class Pointer(pygame.sprite.Sprite):
    
    def __init__(self, startpos, button, screen=Any ):
        super().__init__()
        self.pos = pygame.math.Vector2(startpos)     
        self.image = pygame.image.load('pngs/Hand.png').convert_alpha()           
        self.canvas = screen
        self.rect = pygame.Rect(0,0, 75, 75)
       
        self.color = (255,40,255)
                
    def drawPointer(self): 
        blit_position = self.image.get_rect(center=self.pos)       
        self.canvas.blit(self.image, blit_position)
        #pygame.draw.circle(self.canvas, self.color, self.pos,50)
        
    def update(self, position):
        self.pos = position
        self.rect.center = position
 
class Baskets(pygame.sprite.Sprite):     
    def __init__(self, startpos, signPos,leftHandle,rightHandle, obj_id,rotate = False, screen=Any ):
        super().__init__()
        self.id = obj_id
        self.pos = pygame.math.Vector2(startpos)     
        self.signPos = self.pos + signPos
        self.image = pygame.image.load('pngs/Basket.png').convert_alpha()    
        self.image_answer = pygame.image.load('pngs/SmallSign.png').convert_alpha() 
        self.flip = rotate    
        self.canvas = screen        
        self.left_right_points = [pygame.math.Vector2(leftHandle),pygame.math.Vector2(rightHandle)] 
        
    def draw(self):
       image = pygame.transform.scale(self.image,(250,250))     
       imageFlippedBasket = pygame.transform.flip(image, self.flip, 0)
       blit_position = image.get_rect(center=self.pos)
       
       self.canvas.blit(imageFlippedBasket, blit_position)
       
       image_answer = pygame.transform.scale(self.image_answer,(300,300))
       imageFlipped = pygame.transform.flip(image_answer, self.flip, 0)
       blit_ = imageFlipped.get_rect(center=self.signPos)
       self.canvas.blit(imageFlipped, blit_)
       
    def get_position(self):
       return self.left_right_points

class menuButton(pygame.sprite.Sprite):
	
    def __init__(self, pos, buttonWidth, buttonHeight, buttonText, font, screen):
        super().__init__()
        self.canvas = screen
        self.pos = pygame.math.Vector2(pos) 
        self.buttonWidth = buttonWidth
        self.buttonHeight = buttonHeight
        self.buttonText = buttonText
           
        self.color = (255,255,255)
        self.rect = pygame.Rect(0,0, self.buttonWidth, self.buttonHeight)
        self.rect.center = self.pos
        self.image = pygame.image.load('pngs/BigSign.png').convert_alpha()
        self.textColor = (180, 240, 255)
        self.font = font
       
        self.menuTimer = 0
        
    def collision(self, colliders):
        for rect in colliders:
            colliding = self.rect.colliderect(rect.rect)
            if colliding: 
                
                self.colliding = True
                if self.menuTimer == 0:
                    self.menuTimer = pygame.time.get_ticks()
                    
                elapsed_time = pygame.time.get_ticks() - self.menuTimer  
                self.change_colour(elapsed_time)
                
                if elapsed_time >= 2000:
                    self.menuTimer = 0
                    self.change_colour(0)
                    return True    
            else:
                if self.menuTimer == 0:
                    return                
              
                self.menuTimer = 0
                self.change_colour(0)
                return    

    def draw(self): # Draw the button on screen
        
        image = pygame.transform.scale(self.image,(400,300))
        blit_position = image.get_rect(center=self.pos)     
        image.fill(self.color, special_flags=pygame.BLEND_RGB_MULT)
        self.canvas.blit(image, blit_position)
        
        txt = self.font.render(self.buttonText, True, self.textColor)
        position = txt.get_rect(center=self.pos)
        self.canvas.blit(txt, position)      

    def change_colour(self, timer): # Change the colour of the button when the mouse hovers over it
    	self.color = (255 - (timer*0.05), 255 - (timer*0.05), 255 - (timer*0.05))
        
class Questions():
    def __init__(self, startpos, answersPos,font, color, text, width, JsonData,screen):
        super().__init__()        
        self.font = font
        self.color = color
        self.text = text
        self.width = width
        self.positions = [startpos, answersPos[0], answersPos[1]]
        self.canvas = screen
        self.data = JsonData['Questions']   
        self.image = pygame.image.load('pngs/BigSign.png').convert_alpha() 
        
        self.AnswersDict = []
        self.Counter = 0
        self.LogJsonFile = open('questionlog.json', 'a')


    def update(self, text):
        self.text = text
    
    def wrap_text_q(self, text, width):
        words = text.split(" ")
        lines = []
        current_text = ""
        for word in words:
            tmp_text = current_text + " " + word if current_text else word
            if pygame.font.Font.size(self.font,tmp_text)[0] <= width:
                current_text = tmp_text
            else:
                lines.append(current_text)
                current_text = word   
                
        lines.append(current_text)
        return lines        
                
    def draw(self):
       image = pygame.transform.scale(self.image,(900,500))
       blit_pos = image.get_rect(center=self.positions[0] + (10,0))
       self.canvas.blit(image, blit_pos)
       
       for i,text in enumerate(self.text):
           width = self.width
           if i > 0:
               width = self.width - 400
           lines = self.wrap_text_q(text, width)
           pos = self.positions[i].copy()
           for wrapped in lines:
               txt = self.font.render(wrapped, True, self.color[i])
               question_position = txt.get_rect(center=pos)
               self.canvas.blit(txt, question_position)
               pos.y += pygame.font.Font.size(self.font,wrapped)[1]

    def load_question(self):
        text = self.data[random.randrange(0, len(self.data))]
        self.counter_questions()
        self.update([text['Question'], text['Answer1'], text['Answer2']])
        
    def log_answer(self, choice):
        self.AnswersDict.append({"Question" + str(self.Counter):[
                                {"Question": self.text[0], "Answer":  self.text[choice]}]})
        
        
    def counter_questions(self):
        self.Counter += 1 
        
    def session_end(self, event):
       
        if self.Counter >= 10:
            self.save_session()
            self.Counter = 0
            pygame.event.post(pygame.event.Event(event))
            return False
        else:
            return True
        
    def save_session(self):
        if self.LogJsonFile.closed:
            self.LogJsonFile = open('questionlog.json', 'a')

        NewJson = json.dumps({"Session"+str(datetime.datetime.now().strftime("%H%M%S")): self.AnswersDict}, indent=2, separators=(',', ':'))
        self.LogJsonFile.write(NewJson)
class Text_Draw():
    def __init__(self, startpos, font, color, text, width,final_text,screen):
        super().__init__()        
        self.font = font
        self.color = color
        self.text = text
        self.width = width
        self.position = startpos
        self.text_position = startpos
        self.canvas = screen        
        self.image = pygame.image.load('pngs/BigSign.png').convert_alpha() 
        self.final_text = final_text
        
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
    def updatePosition(self, position, text_position, font):
        self.position = position
        self.text_position = text_position
        self.font = font
        
    def draw(self):
       image = pygame.transform.scale(self.image,(1080,720))
       blit_pos = image.get_rect(center=self.position)
       self.canvas.blit(image, blit_pos)
       
       pos = self.text_position.copy()
       for i,text in enumerate(self.text):
           pos.y += 8
           lines = self.wrap_text_q(text)           
           for wrapped in lines:
               txt = self.font.render(wrapped, True, self.color)
               question_position = txt.get_rect(center=pos)
               self.canvas.blit(txt, question_position)
               pos.y += pygame.font.Font.size(self.font,wrapped)[1]
    

def imagePyGame(screen, frame, text,font):
   
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    frame = pygame.surfarray.make_surface(frame.swapaxes(1,0))
    #frame = pygame.transform.scale(frame, screen.get_size())
    
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(frame, (0,0))
    screen.blit(text_surface, (10, 10))   

def closestPointOnSegment(center, point_a, point_b):

	tangent = point_b - point_a

	if(center - point_a) * tangent <= 0:
		return point_a #'pos' is before 'a' on the line (ab)

	if(center - point_b) * tangent >= 0:
		return point_b # 'pos' is after 'b' on the line (ab)

	# normalize tangent
	T = tangent * (1.0 / pygame.math.Vector2.magnitude(tangent));
	relativePos = center - point_a
	return point_a + T * (T * relativePos)
 
def collideCircleWithSegment(circleCenter, radius,point_a, point_b):

	delta = circleCenter - closestPointOnSegment(circleCenter, point_a, point_b);
	
	if(delta * delta > radius * radius):
		return [0, 0];
	
	dist = pygame.math.Vector2.magnitude(delta)
	N = delta * (1.0 / dist)
	return [radius - dist, N, 0]

def collideWithSegments(pos, colliders):
    earliestCollision = [0, 0, 0]
	
    for collider in colliders:
        points = collider.get_position()
        collision = collideCircleWithSegment(pos, 100,points[0], points[1])
        if(collision[0] > earliestCollision[0]):
            earliestCollision = collision            
            earliestCollision[2] = collider

    return earliestCollision

def circle_box(circle_center, circle_radius, rect = Any):
    closest_x = max(rect.left, min(circle_center[0], rect.right))
    closest_y = max(rect.top, min(circle_center[1], rect.bottom))
    
    distance_squared = (circle_center[0] - closest_x) ** 2 + (circle_center[1] - closest_y) ** 2
    return distance_squared < circle_radius ** 2       