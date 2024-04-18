from pickle import TRUE
import socket
import pygame
import utilsImports as funcs
# # Define server address and port
SERVER_HOST = '192.168.182.98'  # Replace with Raspberry Pi IP address
SERVER_PORT = 8888

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_HOST, SERVER_PORT))
led_on = "on"
led_off = "off"
client_socket.send(led_on.encode())
# Send messages to the server
while True:
   
    message = input("Enter message to send (or type 'exit' to quit): ")
    if message.lower() == 'exit':
        break
    client_socket.send(message.encode())

# Close the connection
client_socket.close()
# class PointerLine(pygame.sprite.Sprite):
    
# 	def __init__(self, startpos, screen):
# 		super().__init__()
# 		self.id = 0
# 		self.pos = pygame.math.Vector2(startpos)     
# 		#self.image = pygame.image.load('pngs/basketv1.png').convert_alpha()           
# 		self.canvas = screen
# 		self.left  = pygame.math.Vector2(0,0)
# 		self.right = pygame.math.Vector2(0,0)
# 		self.color = (255,40,255)		
	                
# 	def drawPointer(self):            
# 		pygame.draw.line(self.canvas, self.color,  self.left, self.right, 4)
		    
# 	def update(self, position):				
# 		self.pos = position
# 		self.left  = self.pos - pygame.math.Vector2(100,0)
# 		self.right = self.pos + pygame.math.Vector2(100,0)
		
# class BaloonTest(pygame.sprite.Sprite):

# 	def __init__(self, startpos, radius, velocity, startdir):
# 		super().__init__()
# 		self.pos = pygame.math.Vector2(startpos)
# 		self.velocity = velocity
# 		self.dir = pygame.math.Vector2(startdir).normalize()
# 		self.radius = radius
# 		self.image = pygame.image.load('pngs/balloonv2.png').convert_alpha()      

# 	def reflect(self, NV):
		
# 		self.dir = self.dir.reflect(pygame.math.Vector2(NV))
     
# 	def update(self, segments):
# 		gravity = 0.001
		
# 		self.dir.y += gravity			
# 		if self.dir.length() > self.velocity:
# 			self.dir.scale_to_length(self.velocity)
		
# 		self.pos += self.dir * self.velocity;
# 		depth = 0
# 		normal = None
# 		for i in range(5):
# 			#callback the ids
# 			collision = collideWithSegments(self.pos, segments)
# 			if(collision[0] > 0):
# 				self.dir = (collision[1])
				
# 				depth = collision[0]
# 				normal = collision[1]
# 				break
# 			depth = collision[0]
# 			normal = collision[1]
			
# 		return [depth, normal]
# 		#self.pos += self.dir * self.velocity
	  
# def closestPointOnSegment(center, point_a, point_b):

# 	tangent = point_b - point_a

# 	if(center - point_a) * tangent <= 0:
# 		return point_a #'pos' is before 'a' on the line (ab)

# 	if(center - point_b) * tangent >= 0:
# 		return point_b # 'pos' is after 'b' on the line (ab)

# 	# normalize tangent
# 	T = tangent * (1.0 / pygame.math.Vector2.magnitude(tangent));
# 	relativePos = center - point_a
# 	return point_a + T * (T * relativePos)
 
# def collideCircleWithSegment(circleCenter, radius,point_a, point_b):

# 	delta = circleCenter - closestPointOnSegment(circleCenter, point_a, point_b);
	
# 	if(delta * delta > radius * radius):
# 		return [0, 0];
	
# 	dist = pygame.math.Vector2.magnitude(delta)
# 	N = delta * (1.0 / dist)
# 	return [radius - dist, N]

# def collideWithSegments(pos, colliders):
# 	earliestCollision = [0, 0]
	
# 	for collider in colliders:
		
# 		collision = collideCircleWithSegment(pos, 50,collider[0], collider[1])
# 		if(collision[0] > earliestCollision[0]):
# 			earliestCollision = collision
# 	return earliestCollision


		
# pygame.init() 

# # Screen properties
# screen_width = 1460
# screen_height = 960
# win = pygame.display.set_mode((screen_width, screen_height)) 
# font = pygame.font.Font(None, 36)
# button = funcs.menuButton((800 * .5, 600 * .5),200, 100, "Start", font, win)
# pointer = PointerLine((800 * .5, 600 * .5), win)
# start, velocity, direction = pygame.math.Vector2(screen_width * .5, 50), 1, (0, 1)
# ballon = BaloonTest(start, 50.,velocity, direction)
# colliders = funcs.Colliders(( 50, screen_height - 150),(0,screen_height - 300),(200,screen_height) ,False,win)
# colliders1 = funcs.Colliders((screen_width - 50, screen_height - 150), (screen_width,screen_height - 300),(screen_width - 200,screen_height),True,win)
# while True:
		
# 	win.fill((32, 32, 32))	
# 	pointer.update(pygame.mouse.get_pos())
# 	pointer.drawPointer()	
# 	check = ballon.update([[pointer.left, pointer.right, pointer.id],[colliders.left, colliders.right,colliders.id], [colliders1.left, colliders1.right,colliders1.id]])
	
# 	if ballon.pos.x <= 0:
# 		ballon.reflect((1,0))
# 	if ballon.pos.x >= (screen_width ):
# 		ballon.reflect((-1, 0))
# 	if ballon.pos.y  <= 0:
# 		ballon.reflect((0, -1))	
# 	ballon.pos.y %= screen_height	
# 	#text = str(check[0]) + "  " + str(check[1])
# 	text_surface = font.render(str(ballon.pos), True, (255, 255, 255))   
# 	win.blit(text_surface, (10, 10))   

# 	win.blit(pygame.transform.scale(ballon.image,(100,100)), ballon.pos)
# 	colliders.draw()
# 	colliders1.draw()
# 	#pygame.draw.line(win,(255, 255, 255),(0,screen_height - 300),(200,screen_height), 3)
# 	#pygame.draw.line(win,(255, 255, 255),(screen_width,screen_height - 300),(screen_width - 200,screen_height), 3)
	
# 	pygame.display.update()

# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:		    
# 			pygame.quit()
# 			exit()
	
