# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:17:02 2022

@author: PRAMILA
"""
import streamlit as st
from __future__ import division # ensures no rounding errors from division involving integers
from math import * # enables use of pi, trig functions, and more.
import pandas as pd # gives us the dataframe concept
pd.options.display.max_columns = 50
pd.options.display.max_rows = 9

from pyhdf import SD
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
#import base64
  
st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>Impact of Aerosols in Solar Power Generation </h1>", unsafe_allow_html=True)  
