import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def preprocess_image(pil_image, target_size=(224, 224)):
    """
    Preprocess an image for the neural network model.
    
    Args:
        pil_image: PIL Image object
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        
        pil_image = pil_image.resize(target_size, Image.LANCZOS)
        
        
        img_array = np.array(pil_image)
        
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        elif img_array.shape[2] == 4:
            
            img_array = img_array[:, :, :3]
        
        
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        
        fallback_img = np.zeros(target_size + (3,), dtype=np.float32)
        return fallback_img

def visualize_tumor(original_img, heatmap, threshold=0.5):
    """
    Visualize the tumor by overlaying a heatmap and drawing a circle around
    the tumor region.
    
    Args:
        original_img: Original image as numpy array
        heatmap: Heatmap highlighting the tumor region
        threshold: Threshold for determining tumor region
        
    Returns:
        Annotated image with tumor highlighted
    """
    try:
        
        img_copy = original_img.copy()
        
        
        if np.max(img_copy) <= 1.0:
            img_copy = (img_copy * 255).astype(np.uint8)
        else:
            img_copy = img_copy.astype(np.uint8)
        
        
        if len(img_copy.shape) == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        elif img_copy.shape[2] == 4:
            img_copy = img_copy[:, :, :3]
        
        
        if len(heatmap.shape) > 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        
        
        if heatmap.dtype != np.float32:
            heatmap = heatmap.astype(np.float32)
            
        if np.max(heatmap) > 1.0:
            heatmap = heatmap / 255.0
        
        
        heatmap_resized = cv2.resize(heatmap, (img_copy.shape[1], img_copy.shape[0]))
        
        
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        
        overlay = cv2.addWeighted(img_copy, 0.7, heatmap_colored, 0.3, 0)
        
        
        binary_map = np.zeros_like(heatmap_uint8)
        binary_map[heatmap_resized > threshold] = 255
        
        
        if len(binary_map.shape) > 2:
            binary_map = cv2.cvtColor(binary_map, cv2.COLOR_RGB2GRAY)
        
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if contours:
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            
            if radius > 5:
                
                cv2.circle(overlay, center, radius, (255, 0, 0), 3)
        
        return overlay
    except Exception as e:
       
        print(f"Error in visualize_tumor: {str(e)}")
        
       
        if np.max(original_img) <= 1.0:
            return_img = (original_img.copy() * 255).astype(np.uint8)
        else:
            return_img = original_img.copy().astype(np.uint8)
            
        if len(return_img.shape) == 2:
            return_img = cv2.cvtColor(return_img, cv2.COLOR_GRAY2RGB)
            
        
        h, w = return_img.shape[:2]
        center = (w//2, h//2)
        size = min(w, h) // 4
        
        
        cv2.line(return_img, 
                (center[0]-size, center[1]-size), 
                (center[0]+size, center[1]+size), 
                (0, 0, 255), 3)
        cv2.line(return_img, 
                (center[0]+size, center[1]-size), 
                (center[0]-size, center[1]+size), 
                (0, 0, 255), 3)
                
        return return_img

def get_tumor_info(tumor_class):
    """
    Get information about a specific tumor type.
    
    Args:
        tumor_class: Index or string key of the tumor class
        
    Returns:
        Dictionary with tumor information
    """
    
    if isinstance(tumor_class, int):
        tumor_class = str(tumor_class)
    
    tumor_info = {
        "0": {
            "name": "No Tumor",
            "description": "No brain tumor detected in the image.",
            "symptoms": [],
            "treatments": [],
            "risk_factors": []
        },
        "1": {
            "name": "Glioma",
            "description": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.",
            "symptoms": [
                "Headache",
                "Nausea and vomiting",
                "Confusion or decline in brain function",
                "Memory loss",
                "Personality changes or irritability",
                "Difficulty with balance",
                "Vision problems",
                "Speech difficulties",
                "Seizures"
            ],
            "treatments": [
                "Surgery",
                "Radiation therapy",
                "Chemotherapy",
                "Targeted drug therapy",
                "Clinical trials"
            ],
            "risk_factors": [
                "Age",
                "Exposure to radiation",
                "Family history of glioma",
                "Genetic syndromes"
            ]
        },
        "2": {
            "name": "Meningioma",
            "description": "A tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign).",
            "symptoms": [
                "Headaches",
                "Seizures",
                "Blurred vision",
                "Weakness in arms or legs",
                "Numbness",
                "Language difficulty",
                "Changes in personality or memory"
            ],
            "treatments": [
                "Observation",
                "Surgery",
                "Radiation therapy",
                "Radiosurgery"
            ],
            "risk_factors": [
                "Female gender",
                "Increasing age",
                "Radiation therapy",
                "Neurofibromatosis type 2",
                "Obesity"
            ]
        },
        "3": {
            "name": "Pituitary",
            "description": "A tumor that develops in the pituitary gland at the base of the brain. Most pituitary tumors are noncancerous (benign).",
            "symptoms": [
                "Headaches",
                "Vision problems",
                "Nausea and vomiting",
                "Hormonal imbalances",
                "Body temperature sensitivity",
                "Mood changes",
                "Fertility issues"
            ],
            "treatments": [
                "Medication to control hormone production",
                "Surgery",
                "Radiation therapy",
                "Hormone replacement"
            ],
            "risk_factors": [
                "Family history",
                "Multiple endocrine neoplasia type 1 (MEN1) syndrome",
                "Carney complex",
                "McCune-Albright syndrome",
                "Familial isolated pituitary adenoma"
            ]
        }
    }
    
    return tumor_info.get(tumor_class, {"name": "Unknown", "description": "Unknown tumor type"})

def prepare_image_for_reinforcement(image_array):
    """
    Prepare an image for reinforcement learning update.
    
    Args:
        image_array: Input image as numpy array
        
    Returns:
        Processed image suitable for model updates
    """
    
    if image_array.shape[0] != 224 or image_array.shape[1] != 224:
        
        image_array = cv2.resize(image_array, (224, 224))
    
    
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    
    if np.max(image_array) > 1.0:
        image_array = image_array / 255.0
        
    return image_array
