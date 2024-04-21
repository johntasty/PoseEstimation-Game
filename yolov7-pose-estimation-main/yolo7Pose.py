import torch
import numpy as np

from utilsImports import cv2
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt

def update(frame,model,mode_eval, device, frame_size):
    orig_image = frame #store frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
    image = letterbox(image, (frame_size), stride=64, auto=True)[0]        
    
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    image = image.to(device)  #convert image data to device
    image = image.half() #convert image to half precision (gpu)
    with torch.no_grad():  #get predictions
        output_data, mode_eval = model(image)
    
    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                0.65,   # Conf. Threshold.
                                0.65, # IoU Threshold.
                                nc=model.yaml['nc'], # Number of classes.
                                nkpt=model.yaml['nkpt'], # Number of keypoints.
                                kpt_label=True,
                                persons=2) # Number of persons.  
    return output_data

