#All the libraries

import cv2 as cv2
import colorsys
import json
from matplotlib import pyplot as plt
import numpy as np
import random
import requests
import skimage.color
import skimage.io
import skimage.transform


#Getting user uploaded image to display
input_img_file = 'Colonic Adenocarcinoma Mitoses-3.png'
org_img = skimage.io.imread(input_img_file)

# fig = plt.figure()
# fig.set_size_inches(10, 7.5)
# plt.title("The original image")
# plt.imshow(org_img)
# plt.show()


# In[2]:


import requests

url = 'http://172.17.0.1:5000/model/predict'

# Submit the nuclei detection request by calling the rest API
def get_nuclei(input_img):
    """
    Takes in input image file path and detects poses.
    """
    files = {'image': ('image',open(input_img,'rb'), 'images/png')}
    result = requests.post(url, files=files).json()
    return result



# preds2 = get_nuclei(input_img_file)
# print(json.dumps(preds2, indent=2))

############################
#client.close()


# In[3]:


# Start nuclear segmenter on local host, not mitoses detector
# Submit the rest request and print out the JSON result

preds = get_nuclei(input_img_file)
#print(json.dumps(preds, indent=2))


# In[4]:


#from core.mask_rcnn.mrcnn.visualize import random_colors, apply_mask

def apply_mask(image, mask, color, alpha=1.0):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * -200,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# Decodes the RLE encoded mask to a binary mask  

def rle_decode(rle, shape):
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def visualize(masks,image):
    figsize=(8, 8) 
    _, ax = plt.subplots(1, figsize=figsize)
        
    colors = random_colors(len(masks))
    
    # Show area outside image boundaries.
    #height, width = image.shape[:2]
    ax.set_ylim(org_img.shape[0])
    ax.set_xlim(org_img.shape[1])
    ax.axis('off')
    #ax.set_title('off')
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(len(masks)):
        mask = masks[i]
        color = colors[i]
        masked_image = apply_mask(masked_image, mask, color)
    ax.imshow(org_img.astype(np.uint8))
    #fig.set_size_inches(org_img.shape[1], org_img.shape[0])
    plt.savefig('masklessmito.png',bbox_inches='tight',pad_inches=0)
    ax.imshow(masked_image.astype(np.uint8))
    #fig.set_size_inches(org_img.shape[1], org_img.shape[0])
    plt.savefig('mito.png',bbox_inches='tight',pad_inches=0)
    #plt.show()
    


#draw image and save


# In[5]:


results = preds["predictions"]
masks = []

# Decodes the RLE encoded masks to the binary masks
for result in results:
    mask = result["mask"]
    masks.append(rle_decode(mask, org_img.shape[0:2]))
# Visualize the detected nuclei on the input image    
mask_img=visualize(masks,org_img)


# In[6]:


import numpy as np
from PIL import Image

img = np.array(Image.open('mito.png'))
#print(type(img))
# <class 'numpy.ndarray'>
#print(img.shape[0])
#Image.fromarray(np.flipud(img)).save('data/dst/lena_np_flipud.jpg')

Image.fromarray(np.fliplr(img)).save('mitolr.png')

#Image.fromarray(np.flip(img, (0, 1))).save('data/dst/lena_np_flip_ud_lr.jpg')


# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

frage = np.array(Image.open(input_img_file))
paramfrage=frage.shape

mrage= np.array(Image.open('mito.png'))
parammrage=mrage.shape
# print(mrage.shape)
xorg2mask=((frage.shape[0])/(mrage.shape[0]))
yorg2mask=((frage.shape[1])/(mrage.shape[1]))



image = cv2.imread('mitolr.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap='gray');

blur = cv2.GaussianBlur(gray, (11,11), 0)
#plt.imshow(blur, cmap='gray')

canny = cv2.Canny(blur, 30, 150, 3)
#plt.imshow(canny, cmap='gray')

dilated = cv2.dilate(canny, (1,1), iterations = 2)
#plt.imshow(dilated, cmap='gray')

(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

#####################################

mydict=[]
for i in cnt:
        M = cv2.moments(i)
        if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
                #cv.putText(image, "center", (cx - 20, cy - 20),
                                #cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        mydict.append(f"x:{round((cx*xorg2mask))} y: {round((cy*yorg2mask))}")
        #print(f"x: {round((cx))} y: {round((cy))}")
#cv.imwrite("image.png", image)   

#print(mydict)
#####################################
no_nuclei=len(cnt)
figss = plt.figure(frameon=False)
figss.set_size_inches(10, 7.5)
figss.savefig('nuclei count.png',bbox_inches='tight',pad_inches=0)
#No axes
ax = plt.Axes(figss, [0., 0., 1., 1.])
ax.set_axis_off()
figss.add_axes(ax)
#draw image and save

plt.imshow(rgb)
plt.savefig('no of nuclei.png', bbox_inches='tight',pad_inches=0)
plt.title("Nuclei in the image: %i" %no_nuclei)
plt.savefig('count of nuclei.png', bbox_inches='tight',pad_inches=0.2)
#plt.show()


# In[8]:


# frage = cv2.imread("Colonic Adenocarcinoma Mitoses-3.png", cv2.IMREAD_UNCHANGED)
# print(frage.shape)
# mrage= cv2.imread("mito.png", cv2.IMREAD_UNCHANGED)
# print(mrage.shape)
# print((frage.shape[0])/(mrage.shape[0]))
# print((frage.shape[1])/(mrage.shape[1]))
# flipHorizontal = cv2.flip(mrage, 1)
# cv2.imshow('Flipped horizontal image', flipHorizontal)


# In[9]:


# from PIL import Image

# old_im = Image.open('Colonic Adenocarcinoma Mitoses-3.png')
# old_size_x = old_im.size[0]
# old_size_y = old_im.size[1]
# new_size = (old_size_x, old_size_y)
# new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
# new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
#                       (new_size[1]-old_size[1])//2))

#new_im.show()
#put in range()
#len(mydict)
# for i in range(len(mydict)):
#     img=Image.open("masklessmito.png")
#     x=int(mydict[i][2:5])
#     y=int(mydict[i][9:12])
#     a=x-32
#     b=y-32
#     c=x+32
#     d=y+32
#     area = (a, b, c, d)
#     cropped_img = img.crop(area)
#     ('nuc' + 'image_' + '_'+ str(i)+'.png', cropped_img)
#     #cropped_img.save("crop_" + str(i) + ".png")
#     cropped_img.show()
#     print(cropped_img.size)
   
    ############################
for i in range(len(mydict)):
    img=Image.open(input_img_file)
    alpha=(mydict[i][2:5])
    alphaRep=alpha.replace('y','')
    x=int(alphaRep)
    beta=(mydict[i][9:12])
    if (beta.count('')==1) and (len(beta)==0):
        betaGood=beta.replace('','0')
        print(betaGood)
        y=int(betaGood)
    else:
        y=int(beta)
    a=x-32
    b=y-32
    c=x+32
    d=y+32
    area = (a, b, c, d)
    cropped_img = img.crop(area)
    ('nuc' + 'image_' + '_'+ str(i)+'.png', cropped_img)
    cropped_img.save("crop_" + str(i) + ".png")
    #cropped_img.show()
    #print(i)
    #print(cropped_img.size)


# In[12]:



import requests
import cv2
from PIL import Image
from paramiko import SSHClient, AutoAddPolicy
import paramiko
from io import StringIO
skey='/root/.ssh/y'
with open(skey) as f:
    fin=f.read()
    
client = SSHClient()
final_key = StringIO(fin)
skeyFile=paramiko.RSAKey.from_private_key(final_key)

client.set_missing_host_key_policy(AutoAddPolicy())
client.connect('###.##.###.##', username='root', pkey=skeyFile)
# stdin,stdout,stderr=client.exec_command('ls')
# output = stdout.readlines()
# for items in output:
#     print(items)


#########################
#Couldn't change port of mitosis detector via config or argument so hosted on different server ip:###.##.###.##



#url = 'http://172.17.0.1:5000/model/predict'
url = 'http://###.##.###.##:5000/model/predict'

# Submit the nuclei detection request by calling the rest API
def get_nuclei(input_img):
    """
    Takes in input image file path and detects poses.
    """
    files = {'image': ('image',open(input_img,'rb'), 'images/png')}
    result = requests.post(url, files=files).json()
    return result



# preds2 = get_nuclei('true.png')
# print(json.dumps(preds2, indent=2))

############################
#client.close()


# In[13]:


probs=[]

#len(mydict)
for i in range(len(mydict)):
    png = Image.open('crop_' + str(i)+ '.png').convert('RGB')
    png.save('crop2rgb_' + str(i)+ '.png', 'PNG', quality=80)
    obj=get_nuclei('crop2rgb_' + str(i)+ '.png')
    pubj=str(obj['predictions'][0])
    qubj=(pubj[15:32])
    rubj=str(pubj[15:32])
    subj=rubj.replace('e','')
    tubj=subj.replace('-','')
    uubj=tubj.replace('}','')
    vubj=float(uubj)
    #print(vubj)
    #print(i)
    probs.append(vubj)
    
##Display these in color for different ranges like red for >1, yellow for <0.5, green for 0.5<x<1.0


# In[14]:


##Let people change threshold before displaying results
fo = open("Python.txt", "r")
uthresh=float(fo.read())
rectcords=[]
finalrects=[]
for j in range(len(probs)):
    if (probs[j]<1.0) and (probs[j]>uthresh):
        finalrects.append([mydict[j][2:5]])
        finalrects.append([mydict[j][9:12]])
        rectcords.append(j)
        #print(probs[j])
 
   
lenfinrec=(len(finalrects)/2)+(len(finalrects)%2)
intlenfin=int(lenfinrec)
#print(intlenfin)
#print("Mito count: %i" %lenfinrec)
#print(rectcords)
#print(finalrects)


# In[15]:


image1 = skimage.io.imread(input_img_file)   

#divide len by 2 append two non-values for -2
for coords in range(intlenfin):
    oddnos=[a for a in range(0,(len(finalrects))) if a%2 != 0][coords]
    y = int(finalrects[oddnos][0])
    evennos=[a for a in range(0,(2*len(finalrects))) if a%2 == 0][coords]
    x = int(finalrects[evennos][0])
    #print(coords)
    #print(evennos)
    #print(oddnos)
    #print(x)
    #print(y)
    #t=round(x/xorg2mask)
    #u=round(y/yorg2mask)
    res=cv2.circle(image1,(x,y), 63, (0,0,255), 1)
    


# In[270]:


mitonum=(len(finalrects)/2)
figfin = plt.figure(frameon=False)
figfin.set_size_inches(10, 7.5)
axfin = plt.Axes(figfin, [0., 0., 1., 1.])
axfin.set_axis_off()
figfin.add_axes(axfin)
plt.imshow(res)
plt.title("Mitoses counted: %i" %mitonum)
figfin.savefig('sunrise.png',bbox_inches='tight',pad_inches=0.2)
#plt.show()




# plt.imshow(rgb)
# plt.savefig('no of nuclei.png', bbox_inches='tight',pad_inches=0)
# 
# plt.show()


# In[ ]:


client = None
try:
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect('###.###.###.###', username='root', password='pass')
    stdin, stdout, stderr = client.exec_command('ls -l')
finally:
    if client:
        client.close()
