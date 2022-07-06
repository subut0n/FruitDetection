import numpy as np 
import pandas as pd 

def return_class_tflite ( image: np.ndarray, detections: list) : # For tflite 
  result_text = []
  for detection in detections:
    # Draw label and score
    category = detection.categories[0]
    probability = round(category.score, 2)
    class_num = category.label
    text = class_num + ' (' + str(probability) + ')'
    result_text.append(text)
  return result_text

def return_class_yolo(): # Return list o string for predictions 
    pred = []
    import os 
    with open("App/A_iFruits/static/yolov5/runs/detect/yolo_foodex/labels/file_upload.txt","r+") as f :
        for i in f.readlines():
            class_num = i.split()[0]
            class_num = query_class_name(class_num)
            operation = f'{class_num}: {i.split()[5]}'
            pred.append(operation)
        f.truncate(0) 
    return pred

def query_class_name(pred):
    for key, value in labels.items():
        if int(pred) == int(value):
            return key 

def return_fun_fact(pred_yolo): # Return list of string for funfact prediction
  df = pd.read_csv('Data/fruits_description.csv')
  labels = [i.split(':')[0] for i in pred_yolo]
  unique = list(set(labels))
  funfact_text = []
  for name in unique : 
      funfact_text.append(f"{name} : {df[df.name == name].description.values[0]}")
  return funfact_text


labels = {'Apple' : 0,  'Apricot' : 1,  'Avocado' : 2,  'Banana' : 3,  'Beetroot' : 4,  'Blueberry' : 5,  'Cactus' : 6,
          'Cantaloupe' : 7, 'Carambula' : 8, 'Cauliflower': 9, 'Cherry' : 10, 'Chestnut' : 11, 'Clementine' : 12, 
          'Cocos' : 13, 'Corn' : 14, 'Dates' : 15, 'Eggplant' : 16, 'Ginger' : 17, 'Granadilla' : 18, 
          'Grape' : 19, 'Grapefruit' : 20, 'Guava' : 21, 'Hazelnut' : 22, 'Huckleberry' : 23, 'Kaki' : 24,
          'Kiwi' : 25,'Kohlrabi' : 26, 'Kumquats' : 27, 'Lemon' : 28, 'Limes' : 29, 'Lychee' : 30,
          'Mandarine' : 31, 'Mango' : 32, 'Mangostan' : 33, 'Maracuja' : 34, 'Melon' : 35, 'Mulberry' : 36,
          'Nectarine' : 37, 'Nut' : 38, 'Onion' : 39, 'Orange' : 40, 'Papaya' : 41, 'Passion' : 42,
          'Peach' : 43, 'Pear' : 44, 'Pepino' : 45, 'Pepper' : 46, 'Physalis' : 47, 'Pineapple' : 48,
          'Pitahaya' : 49, 'Plum' : 50, 'Pomegranate' : 51 , 'Pomelo' : 52, 'Potato' : 53, 'Pumpkin' : 54,
          'Quince' : 55, 'Rambutan' : 56, 'Raspberry' : 57, 'Redcurrant' : 58, 'Salak' : 59, 'Strawberry' : 60,
          'Tamarillo' : 61, 'Tangelo' : 62, 'Tomato' : 63, 'Walnut' : 64}