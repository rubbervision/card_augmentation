import cv2
import numpy as np
import os
import random
import json
from dotenv import load_dotenv

#OTHER

#You want to adjust the values within the augmentation? Here you can add an example grid on your image
def add_grid(image, step=50):
    grid_image = image.copy()
    height, width = image.shape[:2]
    grid_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 255, 255)

    for x in range(0, width, step):
        cv2.line(grid_image, (x, 0), (x, height), grid_color, 1)
        cv2.putText(grid_image, str(x), (x + 5, 15), font, font_scale, text_color, font_thickness)

    for y in range(0, height, step):
        cv2.line(grid_image, (0, y), (width, y), grid_color, 1)
        cv2.putText(grid_image, str(y), (5, y - 5), font, font_scale, text_color, font_thickness)

    return grid_image

#Create a folder if it is not already existing
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

#TRANSFORMATION FUNCTIONS

def transform_card(card, background, x1,y1,x2,y2,x3,y3,x4,y4):
    # Perspective transformation
    (h, w) = card.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([[x1, y1], #Top Left
                            [x2, y2], #Top right
                            [x3, y3], #Bottom right
                            [x4, y4]]) #Bottom Left
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_card = cv2.warpPerspective(card, matrix, (background.shape[1], background.shape[0]))

    # Split into color and alpha channels
    warped_bgr = warped_card[:, :, :3]  # BGR
    warped_alpha = warped_card[:, :, 3]  # Alpha
    return warped_bgr, warped_alpha

def gray_filter(blue_percentage, green_percentage, red_percentage, warped_gray, warped_bgr, warped_alpha):
    
    grayish_card = cv2.merge([
        warped_gray * (1-blue_percentage) + warped_bgr[:, :, 0] * blue_percentage,
        warped_gray * (1-green_percentage) + warped_bgr[:, :, 1] * green_percentage,
        warped_gray * (1-red_percentage) + warped_bgr[:, :, 2] * red_percentage
    ]).astype(np.uint8)

    # Reapply the alpha channel
    grayish_card_with_alpha = cv2.merge([grayish_card[:, :, 0], grayish_card[:, :, 1], grayish_card[:, :, 2], warped_alpha])
    return grayish_card, grayish_card_with_alpha

def apply_gradient_shadow(overlay, direction="vertical", intensity=random.uniform(0.3, 0.6)):
    # Apply a gradient shadow effect to the overlay image.

    # Get dimensions of the overlay
    h, w = overlay.shape[:2]

    # Create the gradient mask
    if direction == "vertical":
        gradient = np.linspace(1 - intensity, 1, h).reshape(-1, 1).astype(np.float32)
        gradient_mask = np.repeat(gradient, w, axis=1)
    elif direction == "horizontal":
        gradient = np.linspace(1 - intensity, 1, w).reshape(1, -1).astype(np.float32)
        gradient_mask = np.repeat(gradient, h, axis=0)
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'")

    # Convert the mask to 3 channels (or 4 if alpha exists)
    if overlay.shape[2] == 4:  # BGRA image
        gradient_mask = np.dstack([gradient_mask] * 3 + [np.ones_like(gradient_mask)])
    else:  # BGR image
        gradient_mask = np.dstack([gradient_mask] * 3)

    # Apply the gradient mask to the overlay
    shadowed_overlay = (overlay.astype(np.float32) * gradient_mask).astype(np.uint8)

    return shadowed_overlay

#TRAIN AUGMENTATIONS

def augmentation_train_1(file_path, output_name, img_dir, label_dir, id, class_id, bg_img):
    # Load images
    background = cv2.imread(bg_img, cv2.IMREAD_UNCHANGED)
    if background is None:
        raise ValueError("Failed to load background image. Check the file path or format.")
    card = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if card is None or card.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (RGBA).")

    x = random.uniform(-100,150)
    y = random.uniform(-100,90)
    x1 = x+random.uniform(235,245)
    y1 = y+random.uniform(165,175)
    x2 = x+random.uniform(410,420)
    y2 = y+random.uniform(225,235)
    x3 = x+random.uniform(330,340)
    y3 = y+random.uniform(480,490)
    x4 = x+random.uniform(135,145)
    y4 = y+random.uniform(410,420)

    warped_bgr, warped_alpha = transform_card(card, background, x1,y1,x2,y2,x3,y3,x4,y4)

    # Convert to grayscale and blend with the original card
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)


    grayish_card, grayish_card_with_alpha = gray_filter(blue_percentage=random.uniform(0.7,0.8),
                                                        green_percentage=random.uniform(0.95,0.99),
                                                        red_percentage=random.uniform(0.92,0.96),
                                                        warped_gray=warped_gray, 
                                                        warped_bgr=warped_bgr, 
                                                        warped_alpha=warped_alpha)

    grayish_card_with_shadow = apply_gradient_shadow(grayish_card, direction="vertical", intensity=0.3)

    # Apply transparency blending
    for c in range(3):  # Loop over B, G, R channels
        background[:, :, c] = background[:, :, c] * (1 - warped_alpha / 255.0) + \
                            grayish_card_with_shadow[:, :, c] * (warped_alpha / 255.0)

    # Add grid to the result
    #result_with_grid = add_grid(background, step=50)

    cv2.imwrite(f"{img_dir}/{output_name}___t1_{id}.jpg", background)

        # Create a YOLO Label
    x_list = [x1,x2,x3,x4]
    y_list = [y1,y2,y3,y4]
    
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    bg_height, bg_width = background.shape[:2]

    x_center = ((x_min + x_max)/2) / bg_height
    y_center = ((y_min + y_max) / 2) / bg_height
    width = (x_max - x_min) / bg_width
    height = (y_max - y_min) / bg_height


    label = str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    with open(f"{label_dir}/{output_name}___t1_{id}.txt","w") as file:
        file.write(str(label))

def augmentation_train_2(file_path, output_name, img_dir, label_dir, id, class_id,bg_img):
    # Load images
    background = cv2.imread(bg_img, cv2.IMREAD_UNCHANGED)
    if background is None:
        raise ValueError("Failed to load background image. Check the file path or format.")
    card = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if card is None or card.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (RGBA).")
    
    x = random.uniform(-50,150)
    y = random.uniform(-100,50)
    x1 = x+random.uniform(145,155)
    y1 = y+random.uniform(295,305)
    x2 = x+random.uniform(335,345)
    y2 = y+random.uniform(300,310)
    x3 = x+random.uniform(340,350)
    y3 = y+random.uniform(550,560)
    x4 = x+random.uniform(145,155)
    y4 = y+random.uniform(545,560)

    # Create a YOLO Label
    x_list = [x1,x2,x3,x4]
    y_list = [y1,y2,y3,y4]
    
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    bg_height, bg_width = background.shape[:2]

    x_center = ((x_min + x_max)/2) / bg_height
    y_center = ((y_min + y_max) / 2) / bg_height
    width = (x_max - x_min) / bg_width
    height = (y_max - y_min) / bg_height


    label = str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    print(label)

    warped_bgr, warped_alpha = transform_card(card, background, x1,y1,x2,y2,x3,y3,x4,y4)

    # Convert to grayscale and blend with the original card
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)


    grayish_card, grayish_card_with_alpha = gray_filter(blue_percentage=random.uniform(0.7,0.8),
                                                        green_percentage=random.uniform(0.95,0.99),
                                                        red_percentage=random.uniform(0.92,0.96),
                                                        warped_gray=warped_gray, 
                                                        warped_bgr=warped_bgr, 
                                                        warped_alpha=warped_alpha)

    grayish_card_with_shadow = apply_gradient_shadow(grayish_card, direction="vertical", intensity=0.3)

    # Apply transparency blending
    for c in range(3):  # Loop over B, G, R channels
        background[:, :, c] = background[:, :, c] * (1 - warped_alpha / 255.0) + \
                            grayish_card_with_shadow[:, :, c] * (warped_alpha / 255.0)

    # Add grid to the result
    #result_with_grid = add_grid(background, step=50)

    cv2.imwrite(f"{img_dir}/{output_name}___t2_{id}.jpg", background)

        # Create a YOLO Label
    x_list = [x1,x2,x3,x4]
    y_list = [y1,y2,y3,y4]
    
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    bg_height, bg_width = background.shape[:2]

    x_center = ((x_min + x_max)/2) / bg_height
    y_center = ((y_min + y_max) / 2) / bg_height
    width = (x_max - x_min) / bg_width
    height = (y_max - y_min) / bg_height


    label = str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    with open(f"{label_dir}/{output_name}___t2_{id}.txt","w") as file:
        file.write(str(label))

#VAL AUGMENTATIONS

def augmentation_val_1(file_path, output_name, img_dir, label_dir, id, class_id, bg_img):
    # Load images
    background = cv2.imread(bg_img, cv2.IMREAD_UNCHANGED)
    if background is None:
        raise ValueError("Failed to load background image. Check the file path or format.")
    card = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if card is None or card.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (RGBA).")
    
    x = random.uniform(-50,150)
    y = random.uniform(-50,100)
    x1 = x+random.uniform(135,145)
    y1 = y+random.uniform(225,235)
    x2 = x+random.uniform(330,340)
    y2 = y+random.uniform(165,175)
    x3 = x+random.uniform(410,420)
    y3 = y+random.uniform(410,420)
    x4 = x+random.uniform(235,245)
    y4 = y+random.uniform(480,490)

    warped_bgr, warped_alpha = transform_card(card, background, x1,y1,x2,y2,x3,y3,x4,y4)

    # Convert to grayscale and blend with the original card
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)


    grayish_card, grayish_card_with_alpha = gray_filter(blue_percentage=random.uniform(0.7,0.8),
                                                        green_percentage=random.uniform(0.95,0.99),
                                                        red_percentage=random.uniform(0.92,0.96),
                                                        warped_gray=warped_gray, 
                                                        warped_bgr=warped_bgr, 
                                                        warped_alpha=warped_alpha)

    grayish_card_with_shadow = apply_gradient_shadow(grayish_card, direction="vertical", intensity=0.3)

    # Apply transparency blending
    for c in range(3):  # Loop over B, G, R channels
        background[:, :, c] = background[:, :, c] * (1 - warped_alpha / 255.0) + \
                            grayish_card_with_shadow[:, :, c] * (warped_alpha / 255.0)


    #Add grid to the result and check image
    # result_with_grid = add_grid(background, step=50)
    # cv2.imshow("Warped Grayscale Card with Transparency", result_with_grid)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(f"{img_dir}/{output_name}___v1_{id}.jpg", background)

    # Create a YOLO Label
    x_list = [x1,x2,x3,x4]
    y_list = [y1,y2,y3,y4]
    
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    bg_height, bg_width = background.shape[:2]

    x_center = ((x_min + x_max)/2) / bg_height
    y_center = ((y_min + y_max) / 2) / bg_height
    width = (x_max - x_min) / bg_width
    height = (y_max - y_min) / bg_height


    label = str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    with open(f"{label_dir}/{output_name}___v1_{id}.txt","w") as file:
        file.write(str(label))



# MAIN

load_dotenv()

image_dir = os.getenv("PATH_TO_CARDS")                   #get root directory
object_list_json = os.getenv("OBJECT_LIST_JSON")    #get json object with list
bg_img = os.getenv("BACKGROUND_IMG")

img_train_folder_name = create_folder("images/train")
img_val_folder_name = create_folder("images/val")
labels_train_folder_name = create_folder("labels/train")
labels_val_folder_name = create_folder("labels/val")

with open(object_list_json ,"r") as file:
    data = json.load(file)

for dirpath, dirnames, filenames in os.walk(image_dir):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename) 
        output_name = filename.split('.')[0]+ '.' + filename.split('.')[1] #Adjust the namimg here as needed
        for index, obj in enumerate(data):
            if obj.get('id') == output_name:
                class_id = obj.get('model_class_id')
       
        for id in range(5): #Adjust the range if you want more iterations per augmentation.
            #train
            augmentation_train_1(file_path,output_name,img_train_folder_name,labels_train_folder_name, id, class_id, bg_img)
            augmentation_train_2(file_path,output_name,img_train_folder_name,labels_train_folder_name, id, class_id, bg_img)
            #val
            augmentation_val_1(file_path,output_name,img_val_folder_name,labels_val_folder_name, id, class_id, bg_img)

        # Show and save the result
        # cv2.imshow("Warped Grayscale Card with Transparency", result_with_grid)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
