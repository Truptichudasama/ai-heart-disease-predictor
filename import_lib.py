import pandas as pd
import numpy as np
import sqlite3
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from flask import Flask, request, jsonify, render_template
import joblib
# Optional: Set a global style for your project
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)

print("Libraries and configurations loaded successfully!")