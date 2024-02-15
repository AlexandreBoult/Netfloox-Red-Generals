from flask import Flask,render_template,request,current_app,send_from_directory,redirect,url_for,flash,Response,jsonify
import os
import pandas as pd
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
import multiprocessing
import sass