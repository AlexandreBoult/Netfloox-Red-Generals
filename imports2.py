from flask import Flask,render_template,request,current_app,send_from_directory,redirect,url_for,flash,Response
import os
import pandas as pd
from Spam_filter_main import train_model,test_msg
import ast
from pathlib import Path
from werkzeug.utils import secure_filename
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import RocCurveDisplay
import random as rd
import shutil
import base64
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
import multiprocessing