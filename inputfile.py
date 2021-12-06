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


st.header("Identify anomalies in chest")
st.write("Choose a Chest X Ray Image")

uploaded_file = st.file_uploader("Choose an image...")







def read_image(imgpath):

    if (str(imgpath).find("jpg")!=-1) or (str(imgpath).find("png")!=-1):

        # sample = Image.open("JPG_test/0c4eb1e1-b801903c-bcebe8a4-3da9cd3c-3b94a27c.jpg")
          sample = Image.open(imgpath)
          return np.array(sample)
    if str(imgpath).find("dcm")!=-1:

        img = dicom.dcmread(imgpath).pixel_array
        # image_rescaled = rescale(img, 0.5, anti_aliasing=False)
        image_resized = resize(img, (img.shape[0] // 2, img.shape[1] // 2),
                       anti_aliasing=True)
        return image_resized
    
         
def generatemodel(xrvmodel,wts):
    # odel = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
    return xrvmodel(weights=wts)
def transform2(img):
    input_tensor = torch.from_numpy(img).unsqueeze(0)
    img = input_tensor.numpy()[0, 0, :]
    img = (img / 1024.0 / 2.0) + 0.5
    img = np.clip(img, 0, 1)
    img = Image.fromarray(np.uint8(img * 255) , 'L')
    return img

def load_image(filename, size=(512,1024)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels
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
def testimage(model,img):
    # with torch.no_grad():
    model.eval()
    out = model(torch.from_numpy(img).unsqueeze(0)).cpu()    
    # out = model(img).cpu()   
    # out = torch.sigmoid(out)

    return {key:value  for (key,value)  in zip(model.pathologies, out.detach().numpy()[0]) if len(key)>2}




def outputprob2(img,pr_model,visimage=True):
    ### Read an image
    img = resize(img, (img.shape[0] // 2, img.shape[1] // 2),
                    anti_aliasing=True)
       
    ### Preprocessmodel
    img_t = transform(img)
    ### Test an image
    return testimage(pr_model,img_t)

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


if uploaded_file is not None:

    ds = dicom.dcmread(uploaded_file)
	
    fig, ax = plt.subplots()
    ax.imshow(ds.pixel_array,cmap="gray")
    
    st.pyplot(fig=fig)



    st.subheader("Probabilities of the ouput")
    model = generatemodel(xrv.models.DenseNet,"densenet121-res224-mimic_ch") ### MIMIC MODEL+
    model.eval()
    
    pr = outputprob2(ds.pixel_array,model) 
    # pr = {k: v for k, v in sorted(pr.items(), key=lambda item: item[1])}
    cnt = 1
    pr = dict( sorted(pr.items(), key=operator.itemgetter(1),reverse=True))
    
    for (key,value) in pr.items():

        st.metric(label=key, value=str(value), delta=str(cnt))
        cnt+=1
#st.write(os.listdir())
#     im = imgGen2(uploaded_file)	
#     st.image(im, caption='ASCII art', use_column_width=True) 	
# # if uploaded_file is not None:
#     src_image = read_image(uploaded_file)
#     # image = Image.open(uploaded_file)	
	
#     st.image(src_image, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    # im = imgGen2(uploaded_file)	
    # st.image(im, caption='ASCII art', use_column_width=True) 	
# model = xrv.models.DenseNet(weights="densenet121-res224-all")

# def asciiart(in_f, SC, GCF,  out_f, color1='black', color2='blue', bgcolor='white'):

#     # The array of ascii symbols from white to black
#     chars = np.asarray(list(' .,:irs?@9B&#'))

#     # Load the fonts and then get the the height and width of a typical symbol 
#     # You can use different fonts here
#     font = ImageFont.load_default()
#     letter_width = font.getsize("x")[0]
#     letter_height = font.getsize("x")[1]

#     WCF = letter_height/letter_width

#     #open the input file
#     img = Image.open(in_f)


#     #Based on the desired output image size, calculate how many ascii letters are needed on the width and height
#     widthByLetter=round(img.size[0]*SC*WCF)
#     heightByLetter = round(img.size[1]*SC)
#     S = (widthByLetter, heightByLetter)

#     #Resize the image based on the symbol width and height
#     img = img.resize(S)
    
#     #Get the RGB color values of each sampled pixel point and convert them to graycolor using the average method.
#     # Refer to https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/ to know about the algorithm
#     img = np.sum(np.asarray(img), axis=2)
    
#     # Normalize the results, enhance and reduce the brightness contrast. 
#     # Map grayscale values to bins of symbols
#     img -= img.min()
#     img = (1.0 - img/img.max())**GCF*(chars.size-1)
    
#     # Generate the ascii art symbols 
#     lines = ("\n".join( ("".join(r) for r in chars[img.astype(int)]) )).split("\n")

#     # Create gradient color bins
#     nbins = len(lines)
#     #colorRange =list(Color(color1).range_to(Color(color2), nbins))

#     #Create an image object, set its width and height
#     newImg_width= letter_width *widthByLetter
#     newImg_height = letter_height * heightByLetter
#     newImg = Image.new("RGBA", (newImg_width, newImg_height), bgcolor)
#     draw = ImageDraw.Draw(newImg)

#     # Print symbols to image
#     leftpadding=0
#     y = 0
#     lineIdx=0
#     for line in lines:
#         color = 'blue'
#         lineIdx +=1

#         draw.text((leftpadding, y), line, '#0000FF', font=font)
#         y += letter_height

#     # Save the image file

#     #out_f = out_f.resize((1280,720))
#     newImg.save(out_f)



# def imgGen2(img1):
#   inputf = img1  # Input image file name

#   SC = 0.1    # pixel sampling rate in width
#   GCF= 2      # contrast adjustment

#   asciiart(inputf, SC, GCF, "results.png")   #default color, black to blue
#   asciiart(inputf, SC, GCF, "results_pink.png","blue","pink")
#   img = Image.open(img1)
#   img2 = Image.open('results.png').resize(img.size)
#   #img2.save('result.png')
#   #img3 = Image.open('results_pink.png').resize(img.size)
#   #img3.save('resultp.png')
#   return img2	


# if uploaded_file is not None:
#     #src_image = load_image(uploaded_file)
#     image = Image.open(uploaded_file)	
	
#     st.image(uploaded_file, caption='Input Image', use_column_width=True)
#     #st.write(os.listdir())
#     im = imgGen2(uploaded_file)	
#     st.image(im, caption='ASCII art', use_column_width=True) 	
    
# name = st.text_input('Name')
# if not name:
#             st.warning('Please input a name.')
#             st.stop()
#             st.success('Thank you for inputting a name.')