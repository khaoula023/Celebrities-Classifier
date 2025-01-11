import os
import sys
from dataclasses import dataclass
import cv2
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

# Modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier