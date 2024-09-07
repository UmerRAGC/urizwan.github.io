import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import subprocess
import sys
import time
import subprocess

cmd="docker run -it -p 5000:5000 max-nucleus-segmenter"
subprocess.Popen(cmd, shell=True)

st.title('Mitosis Counter v1.0')

user_thresh = st.number_input('Pick a mitosis probability value step size 0.0001', min_value=0.0, max_value=1.0, value=0.050, step=0.0001)
image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
i=-1


if st.button('Count Mitoses'):
    if image_file is not None:
        i+=1
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        #st.write(file_details)
        with open(os.path.join("",image_file.name),"wb") as f:
            f.write(image_file.getbuffer())
        filename=str(image_file.name)
        os.rename(r'%s' % filename,r'Colonic Adenocarcinoma Mitoses-3.png')
        st.success("Read File")
        user_thresh_str=str(user_thresh)
        with open("Python.txt", "w+") as f:
            f.write(user_thresh_str)
        st.write("Threshold set to %s" %user_thresh_str)
        exec(open("mitoprog.py").read())
        nuclei = Image.open('pic of nuclei.png')
        image = Image.open('sunrise.png')
        st.image(nuclei)
        st.image(image, caption='Output')
        time.sleep(5)
        os.remove("Colonic Adenocarcinoma Mitoses-3.png")
        os.remove('pic of nuclei.png')
        os.remove('sunrise.png')
        os.remove('Python.txt')
    
        
        


