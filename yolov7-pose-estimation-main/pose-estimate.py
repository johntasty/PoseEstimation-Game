import socket
import torch
import argparse
import numpy as np
#YOLO
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
#

from utilsImports import pygame, cv2
from utilsImports import Any, Thread
import utilsImports as funcs

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="test.mp4",device='cpu',view_img=True,
        save_conf=False,line_thickness = 2,hide_labels=False, hide_conf=True):


    # =pi controll
    SERVER_HOST = '192.168.2.73'  #Pi IP address
    SERVER_PORT = 8000
   
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    connected = False
    # Connect to the server
    # try:
    #     client_socket.connect((SERVER_HOST, SERVER_PORT))
    #     connected = True
    # except Exception as e:
    #     print(f"Error connecting to server: {e}")
    
    led_on = "on"
    led_off = "off"
    
    message_flag = False
    #end region
    device = select_device(opt.device) 
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  
    
    model.half()
   
    startTime = 0
    
    poseNum = 0
    poseCur = 0
    
    #full screen display for upscaling
    cv2.namedWindow("Display1", cv2.WND_PROP_FULLSCREEN) 
    cv2.setWindowProperty("Display1",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    windowWidth  = 1460#cv2.getWindowImageRect("Display1")[2] 
    windowHeight = 960#cv2.getWindowImageRect("Display1")[3] 
    cap = funcs.WebcamStream(640, 512)
    cap.start()  
    
    pygame.init()
    screen = pygame.display.set_mode((windowWidth, windowHeight))
    scale_x = windowWidth / 640
    scale_y = windowHeight / 512  
    up_points = (windowWidth, windowHeight)
    cv2.destroyWindow("Display1")
    
    
    
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0]])
    
    #0 Nose #1 Left Eye #2 Right Eye #3 Left Ear #4 Right Ear #5 Left Shoulder #6 Right Shoulder #7 Left Elbow
    #8 Right Elbow #9 Left Wrist #10 Right Wrist #11 Left Hip #12 Right Hip #13 Left Knee #14 Right Knee #15 Left Ankle #16 Right Ankle
    keypoints_of_interest = [9,10]
    
    #game stuff    
    font = pygame.font.Font(None, 36)
    
    spawn_pos_x = windowWidth * .15
    spawn_pos_y = windowHeight - 50
    boxes = [[(spawn_pos_x, spawn_pos_y) ,(spawn_pos_x * 2, spawn_pos_y)],
             [(spawn_pos_x * 5, spawn_pos_y) ,(spawn_pos_x * 6, spawn_pos_y)]]    
          
    player1 = funcs.Player(screen,0,0, 100, 0 , 0)
    player2 = funcs.Player(screen,0,0, 100, 0 , 0)
    players = [player1, player2]
    
    colliders = [funcs.Colliders((50, windowHeight - 150), pygame.Rect(0,0,0,0),False,screen), funcs.Colliders((windowWidth - 50, windowHeight - 150),pygame.Rect(0,0,0,0), True,screen)]

    start, velocity, direction = pygame.math.Vector2(windowWidth * .1, 50), 2.5, (0, 1)
    ballon = funcs.Baloon(start, 50.,velocity, direction)
    #end
    
    #questions handling
    JsonFile = open('Questions.json')
    JsonData = funcs.json.load(JsonFile)
    question = funcs.load_questions(JsonData)

    question_draw = funcs.Questions(pygame.math.Vector2(windowWidth * .5, 45), font, [[255, 255, 0], [255, 255, 255],[255, 255, 255]], question,650, screen)
    
    #
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = 640  #get video frame width
        
        while(cap.isOpened()): #loop until cap opened or video not complete
            ret, frame = cap.read()  #get frame and success from video capture
            screen.fill([0,0,0])
            if ret:
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                
                image = image.to(device)  #convert image data to device
                image = image.half() #convert image to half precision (gpu)
               
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)


                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.65,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True,
                                            persons=2) # Number of persons.
                
                output = output_to_keypoint(output_data)
                frame = cv2.flip(frame,1)
                poseNum = 0
                
                for i, pose in enumerate(output_data):  # detections per image
                   
                    for c in pose[:, 5].unique():
                        poseNum = (pose[:, 5] == c).sum()  # detections per class
                       
                    if len(output_data):  #check if no pose                                                                        
                        for det_index, (*xyxy, conf) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            
                            kpts = pose[det_index, 6:]   
                            radius = 5
                            num_kpts = len(kpts) // 3
                                                                                    
                            for kid in range(num_kpts):    
                                if kid not in keypoints_of_interest:
                                    continue
                                r, g, b = palette[kid]
                                x_coord, y_coord = 640 -kpts[3 * kid], kpts[3 * kid + 1]
                                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                                    conf = kpts[3 * kid + 2]
                                    if conf < 0.5:
                                        continue
                                    index = (kid - 9)
                                   
                                    boxes[det_index][index] = (int(x_coord * scale_x), int(y_coord * scale_y))
                                    
                                    #cv2.circle(frame, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                           
                #pi control
                if poseNum != poseCur and connected:                    
                    message_flag = False
                    
                if not message_flag and connected:
                    if poseNum > 0:
                        client_socket.send(led_on.encode())
                    else:
                        client_socket.send(led_off.encode())
                    message_flag = True
                    
                poseCur = poseNum
                #end pi
                currentTime = funcs.time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                
                resized_up = cv2.resize(frame, (up_points), interpolation= cv2.INTER_LINEAR)
                
                
                
                text =  "FPS : " + str(int(fps)) + "   People: " + str(int(poseNum))

                #todo boxes should stick to the unique person                            
                funcs.imagePyGame(screen,resized_up,text, font) 
                screen.blit(pygame.transform.scale(ballon.image,(100,100)), ballon.pos)
              
                for i, box in enumerate(boxes):                    
                    players[i].update(box[0],box[1])
                    players[i].drawPlayer()
              
                for basket in colliders:
                    basket.draw()
                
                if ballon.pos.y > windowHeight:
                    question = funcs.load_questions(JsonData)
                    question_draw.update(question)
                    
                ballon.pos.y %= windowHeight
                if ballon.pos.x - 25 <= 50:
                    ballon.reflect((1, 0))
                if ballon.pos.x + 25 >= (windowWidth - 50):
                    ballon.reflect((-1, 0))
                
               
                for collider in colliders:
                    dis = funcs.circle_box(ballon.pos,50,collider.rect)
                    if(dis):
                       ballon.reflect((0, 5))
                       break
                for player in players:
                    
                    center = pygame.math.Vector2(player.pos)
                    dis = player.min_dis / 2
                  
                    col = funcs.circle_line_col(ballon.pos, 50,[player.wristLeft, player.wristRIght],player.pos, dis)
                    if col[0]:
                         ballon.reflect(col[1])
                         break
                ballon.update()        
                    
                
                question_draw.draw()
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        JsonFile.close()	
                        if connected:
                            client_socket.send(led_off.encode())
                        cap.stop()
                        cv2.destroyAllWindows()                        
                        
                        pygame.quit()
                        exit()
              
            
        # cap.stop()  
        # cap.release()
        # cv2.destroyAllWindows()
       


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt


#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
