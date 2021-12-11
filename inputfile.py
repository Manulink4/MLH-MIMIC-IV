import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import os,sys
sys.path.insert(0,"..")
from glob import glob
import torch
import torchvision
import sys
import torch.nn.functional as F
import torchxrayvision as xrv
import pydicom as dicom
import PIL # optional
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
import operator
import mols2grid
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from chembl_webresource_client.new_client import new_client

### Title 
st.markdown("<h1 style='text-align: center;'>Chest Anomaly Identifier</h1>",unsafe_allow_html=True)
### Description
st.markdown("""<p style='text-align: center;'>The goal of this application is mainly to help doctors to interpret  
Chest X-Ray Images, being able to find medical compounds in a quick way to deal with Chest's anomalies found</p>""",unsafe_allow_html=True)

### Image
st.image("doctors.jpg")

### Uploder 
# st.markdown("""<p style='text-align: center;'>The goal of this application is mainly to help doctors to interpret  
# Chest X-Ray Images, being able to find medical compounds in a quick way to deal with Chest's anomalies found</p>""",unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an X-Ray image to detect anomalies of the chest (the file must be a dicom extension or jpg)")




#### Get Compounds found
@st.cache(allow_output_mutation=True)
def getdrugs(name,phase):
    drug_indication = new_client.drug_indication
    molecules = new_client.molecule
    obj = drug_indication.filter(efo_term__icontains=name)   
    appdrugs = molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in obj])
    

    if phase!=[]:
        temp = None
        for ph in phase:
            dftemp = pd.DataFrame.from_dict(appdrugs.filter(max_phase=int(ph)))
            dftemp["phase"] = int(ph)
            if isinstance(temp,pd.DataFrame):
                temp= pd.concat([temp,dftemp],axis=0)
            else:
                temp = dftemp

        df = temp
    else:
        df = pd.DataFrame.from_dict(appdrugs)
            
    try:
                df.dropna(subset=["molecule_properties","molecule_structures"],inplace=True)
                
                df["smiles"] = df.molecule_structures.apply(lambda x:x["canonical_smiles"])
                df["Acceptors"] = df.molecule_properties.apply(lambda x :x["hba"])
                df["Donnors"] = df.molecule_properties.apply(lambda x :x["hbd"])
                df["mol_weight"] = df.molecule_properties.apply(lambda x :x["mw_freebase"])
                df["Logp"] = df.molecule_properties.apply(lambda x :x["cx_logp"])

                subs = ["pref_name","smiles","Acceptors","Donnors","mol_weight","Logp"]
                df.dropna(subset=subs,inplace=True)
                df["Acceptors"] =  df["Acceptors"].astype(int)
                df["Donnors"] = df["Donnors"].astype(int)
                df["mol_weight"] =  df["mol_weight"].astype(float)
                df["Logp"] = df["Logp"] .astype(float)
                
                return df.loc[:,subs]
    except:
                return None

### Read Chest X Ray Image
def read_image(imgpath):

    if (str(imgpath).find("jpg")!=-1) or (str(imgpath).find("png")!=-1):

        # sample = Image.open("JPG_test/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
          sample = Image.open(imgpath)
          return np.array(sample)
    if str(imgpath).find("dcm")!=-1:
        img = dicom.dcmread(imgpath).pixel_array
        return img
    
### Generate torchxrayvision model to find output probabilities       
def generatemodel(xrvmodel,wts):
    return xrvmodel(weights=wts)
### Transform the image to ouput some illness
def transform2(img):
    input_tensor = torch.from_numpy(img).unsqueeze(0)
    img = input_tensor.numpy()[0, 0, :]
    img = (img / 1024.0 / 2.0) + 0.5
    img = np.clip(img, 0, 1)
    img = Image.fromarray(np.uint8(img * 255) , 'L')
    return img
### Transform the image to test an output image
def transform(img):

            img = ((img-img.min())/(img.max()-img.min())*255)


            # img = (img / 1024.0 / 2.0) + 0.5
            # img = np.clip(img, 0, 1)
            # img = Image.fromarray(np.uint8(img * 255) , 'L')
            # print(img.shape)
            # img = skimage.io.imread("JPG_test/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
            # print(img.max())
            img = xrv.datasets.normalize(np.array(img), 255) 
            
            # Check that images are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("error, dimension lower than 2 for image")

            # Add color channel
            img = img[None, :, :]

            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                        xrv.datasets.XRayResizer(224,engine="cv2")])

            img = transform(img)
            return img
### Returns the output probabilities of having certain illnesses anomalies
def testimage(model,img):
    # with torch.no_grad():
    model.eval()
    out = model(torch.from_numpy(img).unsqueeze(0)).cpu()    
    # out = model(img).cpu()   
    # out = torch.sigmoid(out)

    return {key:value  for (key,value)  in zip(model.pathologies, out.detach().numpy()[0]) if len(key)>2}

### Resize the model
def outputprob2(img,pr_model,visimage=True):
    ### Read an image
    img = resize(img, (img.shape[0] // 2, img.shape[1] // 2),
                    anti_aliasing=True)
       
    ### Preprocessmodel
    img_t = transform(img)
    ### Test an image
    return testimage(pr_model,img_t)

### Pipeline since we read an image until the ouput it is generated
def outputprob(imgpath,pr_model,visimage=True):
    ### Read an image
    img = read_image(imgpath)
    if visimage:
        plt.imshow(img,cmap="gray")
        plt.show()
    ### Preprocessmodel
    img_t = transform(img)
    ### Test an image
    return testimage(pr_model,img_t)


### Error in case we do not find compounds
def error(option):
    option = str(option).replace(" ","%20")
    st.markdown(f"""
            We have not found compounds for this illness; for more information visit this link:
            [ChEMBL](https://www.ebi.ac.uk/chembl/g/#search_results/all/query={option})
               """, unsafe_allow_html=True)

### If you insert an image
if uploaded_file is not None:

    #### Read an image

   
    imgdef = read_image(uploaded_file)
else:
    imgdef = read_image("example.dcm")
## Controller header

st.sidebar.markdown("<h1 style='text-align: center;'>Compound's filter</h1>",unsafe_allow_html=True)
## Write the compound 
st.sidebar.markdown('''
                <h4 style='text-align: center;'>This controller sidebar is used to filter the compounds by the following features</h4>
                
                - Molecular weight : is the weight of a compound in grame per mol
                - LogP             : it measures how hydrophilic or hydrophobic  a compound is
                - NumDonnors       : number of chemical components that are able to deliver electrons to other chemical components
                - NumAcceptors       : number of chemical components that are able to accept electrons to other chemical components
                ''',unsafe_allow_html=True)
weight_cutoff = st.sidebar.slider(
    label="Molecular weight",
    min_value=0,
    max_value=1000,
    value=500,
    step=10,
    help="Look for compounds that have less or equal molecular weight than the value selected"
)
logp_cutoff = st.sidebar.slider(
    label="LogP",
    min_value=-10,
    max_value=10,
    value=5,
    step=1,
    help="Look for compounds that have less or equal logp than the value selected"
)
NumHDonors_cutoff = st.sidebar.slider(
    label="NumHDonors",
    min_value=0,
    max_value=15,
    value=5,
    step=1,
    help="Look for compounds that have less or equal donors weight than the value selected"
)
NumHAcceptors_cutoff = st.sidebar.slider(
    label="NumHAcceptors",
    min_value=0,
    max_value=20,
    value=10,
    step=1,
    help="Look for compounds that have less or equal acceptors weight than the value selected"
)
max_phase = st.sidebar.multiselect("Phase of the compound",
        ['1','2', '3', '4'],
        help=""" 
        - Phase 1 : Phase I   of the compound in progress
        - Phase 2 : Phase II  of the compound in progress
        - Phase 3 : Phase III of the compound in progress
        - Phase 4 : Approved compound 
        """
    )


### Plot the input image
fig, ax = plt.subplots()
ax.imshow(imgdef,cmap="gray")
st.pyplot(fig=fig)
# Printing the possibility of having anomalies
st.markdown("<h3 style='text-align: center;'>Possibility of anomalies</h3>",unsafe_allow_html=True)
model = generatemodel(xrv.models.DenseNet,"densenet121-res224-mimic_ch") ### MIMIC MODEL+
model.eval()
pr = outputprob2(imgdef,model) 

# Sort results by the descending probability order 
pr = dict( sorted(pr.items(), key=operator.itemgetter(1),reverse=True))
# Select the treatment 
option = st.sidebar.selectbox('Anomaly',list(pr.keys()),help='Select the illness or anomaly you want to treat')
col1,col2,col3 = st.columns((1,1,1))
cnt = 1
for (key,value) in pr.items():
    if cnt%3==1:
        col1.metric(label=key, value=str(cnt), delta=str(value))
    if cnt%3==2:
        col2.metric(label=key, value=str(cnt), delta=str(value))
    if cnt%3==0:
        col3.metric(label=key, value=str(cnt), delta=str(value))
    cnt+=1
    # temp = st.expander("Compunds to take care of {}".format(key))
#### Get the compounds for the anomaly selected
df = getdrugs(option,max_phase)
st.markdown("<h3 style='text-align: center;'>Compounds for {}</h3>".format(option),unsafe_allow_html=True)
### If exists the compounds
if df is not None:
            
            #### Filter dataframe by controllers
            df_result = df[df["mol_weight"] < weight_cutoff]
            df_result2 = df_result[df_result["Logp"] < logp_cutoff]
            df_result3 = df_result2[df_result2["Donnors"] < NumHDonors_cutoff]
            df_result4 = df_result3[df_result3["Acceptors"] < NumHAcceptors_cutoff]
        

            
            if len(df_result4)==0:
                
                error(option)
            else:
                    raw_html = mols2grid.display(df_result,  mapping={"smiles": "SMILES","pref_name":"Name","Acceptors":"Acceptors","Donnors":"Donnors","Logp":"Logp","mol_weight":"mol_weight"},
                    subset=["img","Name"],tooltip=["Name","Acceptors","Donnors","Logp","mol_weight"],tooltip_placement="top",tooltip_trigger="click hover")._repr_html_()
        
                    components.html(raw_html, width=900, height=900, scrolling=True)
#### We do not find compounds for the anomaly 
else:
    error(option)

