### FRAMEWORKS AND DEPENDENCIES
import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from PIL import Image, ImageFilter
from collections import OrderedDict
import matplotlib as mpl
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
# Other methods available: ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.io import imread
import pydicom as dicom
import operator
import mols2grid
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from chembl_webresource_client.new_client import new_client
import streamlit as st

####UTILS.PY
model_names = ['densenet121-res224-mimic_nb', 'densenet121-res224-mimic_ch']

#### FUNCTIONS FOR STREAMLIT
### Cache Drugs (Get Compounds found)
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
### Title 
def header():
    
        st.markdown("<h1 style='text-align: center;'>Chest Anomaly Identifier</h1>",unsafe_allow_html=True)
        ### Description
        st.markdown("""<p style='text-align: center;'>This is a pocket application that is mainly focused on aiding medical 
professionals on their diagnostics and treatments for chest anomalies based on chest X-Rays. On this application, users 
can upload a chest X-Ray image and a deep learning model will output the probability of 14 different anomalies taking 
 place on that image</p>""",unsafe_allow_html=True)

        ### Image
        st.image("doctors.jpg")
### Controllers 
def controllers2(model_probs):
    
    


    # Select the anomaly to detect 
    st.sidebar.markdown("<h1 style='text-align: center;'>Anomaly detection</h1>",unsafe_allow_html=True)
    option_anomaly = st.sidebar.selectbox('Select Anomaly to detect',['Atelectasis', 'Consolidation', 'Pneumothorax','Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly'],help='Select the anomaly you want to detect')
    # Filtering anomalies
    st.sidebar.markdown('''
                    <h4 style='text-align: center;'>This controller is used to filter anomaly detection </h4>
                    
                    - N                : Select the number of most likely anomalies you want to detect
                    - Threshold        : It measures how strict you are with the threshold
                    - Colors           : For color intensity of anomaly detection
                    - Obscureness      : For darker or lighter colors
                    
                  
                    ''',unsafe_allow_html=True)

    N      = st.sidebar.slider(label="N",min_value=1,max_value=5,value=3,step=1,help="Select the number of most likely anomalies you want to detect")      
    threshold      = st.sidebar.slider(label="Threshold",min_value=0.0,max_value=1.0,value=0.3,step=0.1,help="Select the degree of confidence you want to detect. The more is the value the more strict you are in your detection")
    colors      = st.sidebar.slider("Intense Colors",min_value=0.0,max_value=1.0,value=0.6,step=0.1,help="Select the color intensity you want to display at the time on detecting an anomaly. The higuer the value, the more intense the color")
    obscureness      = st.sidebar.slider("Obscureness",min_value=0.0,max_value=1.0,value=0.8,step=0.1,help="Select the obscureness you want your colors have. The higuer the value, the more obscure is the color")
    
    
    # Select the treatment

    st.sidebar.markdown("<h1 style='text-align: center;'>Anomaly Treatment</h1>",unsafe_allow_html=True)
    option = st.sidebar.selectbox('Select the anomaly for treatment',list(model_probs[model_names[0]].keys()),help='Select the anomaly you want to treat')
    
    

    #### Filtering treatments
    st.sidebar.markdown("<h1 style='text-align: center;'>Compound's filter</h1>",unsafe_allow_html=True)
    ## Write the compound 
    st.sidebar.markdown('''
                    <h4 style='text-align: center;'>This controller sidebar is used to filter the compounds by the following features</h4>
                    
                    - Molecular weight : is the weight of a compound in grame per mol
                    - LogP             : it measures how hydrophilic or hydrophobic  a compound is
                    - NumDonnors       : number of chemical components that are able to deliver electrons to other chemical components
                    - NumAcceptors     : number of chemical components that are able to accept electrons to other chemical components
                    - MaxPhase         : select the phase in which the compound is stablished
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
    max_phase = st.sidebar.multiselect("Select Phase of the compound",
            ['1','2', '3', '4'],
            help=""" 
            - Phase 1 : Phase I   of the compound in progress
            - Phase 2 : Phase II  of the compound in progress
            - Phase 3 : Phase III of the compound in progress
            - Phase 4 : Approved compound          """
        )
    
    return option_anomaly,threshold,colors,obscureness,option,weight_cutoff,logp_cutoff,NumHDonors_cutoff,NumHAcceptors_cutoff,max_phase,N

    

### MODEL.PY

def takemodel(models:OrderedDict,cams:OrderedDict,weights="mimic_ch"):
    """
        Define models and cams of each model; tools useful for heatmap
    Args:
        models (OrderedDict[xrv.models.DenseNet]): the CNN of the model
        cams (OrderedDict[GradCam]): Useful tool to make the heatmap
        weights (str): Name of the pretrained model weights
    """
    models[weights] = xrv.models.DenseNet(weights=weights)
    models[weights].eval()
    target_layer = models[weights].features[-2]
    cams[weights] = GradCAM(models[weights], target_layer, use_cuda=False)
    return models,cams
#### Read the image | Normalize
def normalize(sample, maxval):
    """
    Scales images to be roughly [-1024 1024].
     Args:
        image (dicom,jp,png): image
        maxval (int): maxvalue of the dicom image
        
    From torchxrayvision
    """
    
    if sample.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(sample.max(), maxval))
    
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

def extensionimages(image_path):
    """
    Read Image of jpg dicom or png if it does not find the image returns skimage.io.imread(imgpath)
    Args:
    image_path (str): path of the image

    """

    if (str(image_path).find("jpg")!=-1) or (str(image_path).find("png")!=-1):

        # sample = Image.open("JPG_test/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
          sample = Image.open(image_path)
          return np.array(sample)
    if str(image_path).find("dcm")!=-1:
        img = dicom.dcmread(image_path).pixel_array
       
        return img
    else:
        return imread(image_path)


def read_image(img, tr=None,visualize=True):
    """
    Scales images to be roughly [-1024 1024].
     Args:
        image_path (str): path of the image    
    From torchxrayvision
    """
    # img = extensionimages(image_path)
    ### If black image has 3 dim get just one channel
    

    try:
        img = img[:, :, 0]
    ### Otherwise we take 2 channels
    except IndexError:
        pass
    # Another option will be equalizing the image
    # img = cv2.equalizeHist(img.astype(np.uint8))
    img = ((img-img.min())/(img.max()-img.min())*255)
    ### Normalize to values -1024 1024
    img = normalize(img, 255)
    # print(img.min(),img.max())
    # Add color channel
    img = img[None, :, :]
    if tr is not None:                    
        img = tr(img)
    else:
        raise Exception("You should pass a transformer to downsample the images")
    return img

#### Applly colormap on image
def apply_colormap_on_image(org_im, activation, colormap_name, threshold=0.3,alpha=0.6):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image (224x224)
        activation_map (numpy arr): Activation map (grayscale) 0-255 (224x224)
        colormap_name (str): Name of the colormap (colormap_name)
        threshold (float): threshold at which to overlay heatmap (threshold that anomaly must surpass in terms of probability)
        alpha (float): adjust the intense in which the model predicts
    Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations

    Added thresholding to activations.
    """
    ### Grayscale_cam
    grayscale_cam = copy.deepcopy(activation)
    # Get colormap just color type
    color_map = mpl_color_map.get_cmap(colormap_name)
    # Like map the activation function to the color map
    
    no_trans_heatmap = color_map(activation)
    ### Not_trans_heatmap output (224x224x4 channels) (HSV-alpha channels)
    ### H --> channel 0 H --> channel 1 H --> channel 2 alpha --> channel 3
    
    # Change alpha channel in colormap to make sure original image is displayed deepcopy
    alpha_channel = 3
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, alpha_channel] = alpha

    # set to fully transparent if there is a very low activation (if the activation map is lower than the threshold)
    idx = (grayscale_cam <= threshold)
    # convert to a 3d index the shape of the image (expand the image by arrays) 
    # Input shape 224x244 --- Output Shape 224x224x1
    ignore_idx = np.expand_dims(np.zeros(grayscale_cam.shape, dtype=bool), 2)
 
    ### Idx is the four fimenation of the heatmap concatenate 224x224x3 with 224x224x1 ---> 224x224x4
    idx = np.concatenate([ignore_idx]*3 + [np.expand_dims(idx, 2)], axis=2)
    
    
    heatmap[idx] = 0
    ### Inputs 224x224x4 
    ### Scale to a 255 integer and map to PIL image
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    ### Color map activation scale to 255 PIL image
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    
    # Apply heatmap on image
    ### Create and RGBA image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    ### org_im PIL converted onto RGBA and overlapped with heatmap on image
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    ### heatmap_on_image overlap with heatmap
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image   



def heatmap_core(image:np.array,pathologies:list,target:str,model_cmaps:list,threshold = 0.3, alpha = 0.8,obscureness = 0.8,fontsize=14)->plt:
    """
    Returns the heatmap of the image
     Args:
        image (np.array): Numpy Array Image (224x224)
        target (str): Pathology to select
        model_cmaps (list): colors to heatmap
        pathologies(list): List of pathologies
        threshold (float): Threshold to be more exigent or less exigent with the zone in which you are looking for
        alpha (float): the higher this value, the more intense is the colormaps
        obscureness (float) : the mhigher is this value the darker are the color maps
        fontsize (float): adjust the fontsize of the plot
    Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations
    Modifications by : ### TeamMIMICIV  

    Added thresholding to activations.
    """
    
    #### Initializing models 
    models = OrderedDict()
    cams = OrderedDict()
    for model_name in ['densenet121-res224-mimic_nb', 'densenet121-res224-mimic_ch']:
      #### Adding the models and cams to the OrderedDict structure
      models,cams = takemodel(models,cams,weights=model_name)
    ### Get an image
    input_tensor = torch.from_numpy(image).unsqueeze(0)

    img = input_tensor.numpy()[0, 0, :, :]
    img = (img / 1024.0 / 2.0) + 0.5
    img = np.clip(img, 0, 1)
    img = Image.fromarray(np.uint8(img * 255) , 'L')

    # using the variable axs for multiple Axes
    plt.figure(figsize=(10, 8))
    
    i = 0
    for model_name, model in models.items():
      # get our model performance
      with torch.no_grad():
          out = model(input_tensor).cpu()
        
      # reshape the dataset labels to match our model
      # xrv.datasets.relabel_dataset(model.pathologies, d_pc)

      # finds the index of the target based on the model pathologies
      assert target  in pathologies,"Pathology input not in pathology maps"
      target_category = model.pathologies.index(target)
      grayscale_cam = cams[model_name](input_tensor=input_tensor, target_category=target_category)
      # In this example grayscale_cam has only one image in the batch:
      grayscale_cam = grayscale_cam[0, :]

      _, img = apply_colormap_on_image(img, grayscale_cam, model_cmaps[i].name, threshold=threshold,alpha=alpha)

      # add plot to add the color to the axis
      plt.plot(0, 0, '-', lw=6, color=model_cmaps[i](0.7), label=model_name)

      # what did we predict?
      prob = np.round(out[0].detach().numpy()[target_category], 4)
  
      i += 1

    plt.legend(fontsize=fontsize)
    plt.imshow(img, cmap='bone')
    plt.axis('off')
    # plt.show()
    return plt


def heatmap(img,target,threshold = 0.3, alpha = 0.8,obscureness = 0.8,fontsize=14):
    """
    Returns the heatmap of the image
     Args:
        imgpath (str): Name of the image path 
        target (str): Pathology to select
       
        threshold (float): Threshold to be more exigent or less exigent with the zone in which you are looking for
        alpha (float): the higher this value, the more intense is the colormaps
        obscureness (float) : the mhigher is this value the darker are the color maps
        fontsize (float): adjust the fontsize of the plot
    Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations
    Modifications by : ### TeamMIMICIV  
    Added thresholding to activations.
    """
    pathologies = ['Atelectasis', 'Consolidation', 'Pneumothorax','Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly']
    model_cmaps = [mpl_color_map.Purples, mpl_color_map.Greens_r]
    tr = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224, engine='cv2')]
    )
    image = read_image(img,tr=tr)
    return heatmap_core(image,pathologies,target,model_cmaps,threshold = threshold, alpha = alpha,obscureness = obscureness,fontsize=fontsize)


#### Initializing models 
def probtemp(image:np.array)->dict:
    """
    Returns the output probabilities of two models
     Args:
        image (np.array): Numpy already scaled 
    """
    #### Initializing models 
    models = OrderedDict()
    cams = OrderedDict()
    
    for model_name in ['densenet121-res224-mimic_nb', 'densenet121-res224-mimic_ch']:
      #### Adding the models and cams to the OrderedDict structure
      models,cams = takemodel(models,cams,weights=model_name)
    ### Get an image
    input_tensor = torch.from_numpy(image).unsqueeze(0)

    img = input_tensor.numpy()[0, 0, :, :]
    img = (img / 1024.0 / 2.0) + 0.5
    img = np.clip(img, 0, 1)
    img = Image.fromarray(np.uint8(img * 255) , 'L')

    model_dics = {}
    for model_name, model in models.items():
      # get our model performance
      with torch.no_grad():
          out = model(input_tensor).cpu()
          model_dics[model_name] = {key:value  for (key,value)  in zip(model.pathologies, out.detach().numpy()[0]) if len(key)>2}
    return model_dics
def getprobs(img):
    """
    Returns the heatmap of the image
     Args:
        imgpath (str): Name of the image path 
        target (str): Pathology to select
       
        threshold (float): Threshold to be more exigent or less exigent with the zone in which you are looking for
        alpha (float): the higher this value, the more intense is the colormaps
        obscureness (float) : the mhigher is this value the darker are the color maps
        fontsize (float): adjust the fontsize of the plot
    Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations
    Modifications by : ### TeamMIMICIV  
    Added thresholding to activations.
    """
    pathologies = ['Atelectasis', 'Consolidation', 'Pneumothorax','Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly']
    tr = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224, engine='cv2')]
    )
    image = read_image(img,tr=tr)
    return probtemp(image)




#### MORE FUNCTIONS.PY
### Get the probability of models
def sortedmodels(probs,model_name):
            """
            Sorts the probability model 
            Args:
            probs (dict) : dictionary of model probabilities
            model_name (str) : name of the model
            """
            ### Probability of the model
            promodels = probs[model_name]
            # Sort results by the descending probability order 
            return dict(sorted(promodels.items(), key=operator.itemgetter(1),reverse=True))
def disprobs(model_probs,model_name,N):
    """        
            Displays the probability models and Sorts the probability model 
            Args:
            model_probs (dict) : dictionary of model probabilities
            model_name (str) : name of the model
            """
    exp1 = st.expander(f"Probabilities for {model_name}")
    pr = sortedmodels(model_probs,model_name)
    for cnt,(key,value) in enumerate(pr.items()):
         if cnt==N:
             break
         exp1.metric(label=key, value=str(cnt+1), delta=str(value))

def getfile(uploaded_file=None):
    """
    Get the file uploaded
    """
    if uploaded_file is not None:
        return extensionimages(uploaded_file)
    return extensionimages("example.dcm")
### Error in case we do not find compounds
def error(option):
    option = str(option).replace(" ","%20")
    par3 = f'https://www.ebi.ac.uk/chembl/g/#search_results/all/query={option})'
    par2 =  "<a href = {} >".format(par3)
    par =par2 +"ChEBML" + "</a>"
    
    st.markdown("<p style='text-align: center;'>We have not found compounds for this illness; for more information visit this link: {}</p>".format(par), unsafe_allow_html=True)

def main():

            sys.path.insert(0,"..")
            ### Title
            st.set_page_config(layout="wide")
            header()
            ### Uploader  
            uploaded_file = st.file_uploader("Choose an X-Ray image to detect anomalies of the chest (the file must be a dicom extension or jpg)",)
            #### Get the image

            imgdef = getfile(uploaded_file)
            __,col4,_,col5,_,col6,__ = st.columns((0.1,1,0.2,2.5,0.2,1,0.1)) 
            col5.markdown("<h3 style='text-align: center;'>Input Image</h3>",unsafe_allow_html=True)
            with col5:
                ### Plot the input image
                fig, ax = plt.subplots()
                ax.imshow(imgdef,cmap="gray")
                st.pyplot(fig=fig)
            # Printing the possibility of having anomalies

            __,col1,_,col3,_,col2,__ = st.columns((0.1,1,0.2,2.5,0.2,1,0.1)) 
            col3.markdown("<h3 style='text-align: center;'>Anomaly Detection</h3>",unsafe_allow_html=True)
            model_probs  = getprobs(imgdef)
            option_anomaly,threshold,colors,obscureness,option,weight_cutoff,logp_cutoff,NumHDonors_cutoff,NumHAcceptors_cutoff,max_phase,N = controllers2(model_probs)
            ### MODEL 1
            with col1:
                disprobs(model_probs,model_names[0],N)
            ### MODEL_2
            with col2:
                disprobs(model_probs,model_names[1],N)

            ### ANOMALY HEATMAP
            with col3:
                plot = heatmap(imgdef,option_anomaly,threshold,colors,obscureness,14)
                st.pyplot(plot)
            df = getdrugs(option,max_phase)

            st.markdown("<h3 style='text-align: center;'>Compounds for {}</h3>".format(option),unsafe_allow_html=True)
            __,col10,col11,_,_,col12,__ = st.columns((0.1,0.8,2.5,0.2,0.2,1,0.1)) 

            ### TREATMENT FILTERING
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
                                with col11:

                                    components.html(raw_html, width=900, height=900, scrolling=True)
            #### We do not find compounds for the anomaly 
            else:
                
                error(option)

if __name__=="__main__":
    main()