import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

data_path = "/content/traffic-sign-dataset-classification.zip"

with ZipFile(data_path, "r") as zip_ref:
    zip_ref.extractall("/content")

print("Dataset extracted")

dataset = "/content/traffic_Data/DATA"

labelfile = pd.read_csv("/content/labels.csv")

img = cv2.imread("/content/traffic_Data/DATA/10/010_0011.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()


img = cv2.imread("/content/traffic_Data/DATA/23/023_0001.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()

labelfile.head()
labelfile.tail()
