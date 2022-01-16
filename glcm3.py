a=1
#This program classifies images using SIFT/GLCM feature extraction and RF/SVM/LGBM  classifier
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import random

PLOT=1
modeln      = 1             # 1-RF, 2-SVM, 3-LGB
FEATUREEXT  = 2             # 1- SIFT, 2-GLCM
SIZE        = 16            # resize images to 64x64
NUMDIST     = 9             # MAX distance value
DISTSTEP    = 2             # distance change from 0
ANGLESTEP   = np.pi/4       # angle change
NUMANGLES   = 3             # number of angles
path        = "Tomato/"      # folder-> training(class A,class B,class C) and testing(class A,class B,class C)
imagetype   = "*.jpg"       # image format

def SIFT_extractor(dataset):
    kps =np.zeros(dataset.shape[0])
    image_dataset = pd.DataFrame()
    sift = cv2.xfeatures2d.SIFT_create()
    for image in range(dataset.shape[0]):  
        print(str(image+1) +'/'+str(dataset.shape[0]))
        img = dataset[image,:,:]
        kp, des = sift.detectAndCompute(img,None)
        kps[image]= len(kp)       
        des = np.array(des)
        for i in range(0,des.shape[0]):
            df = pd.DataFrame()
            for j in range(0,des.shape[1]):
                df['S'+str(format(j,'03d'))] = [des[i,j]]                
            image_dataset = image_dataset.append(df)
    kps=kps.astype(np.int64)
    return image_dataset,kps

def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):        
        df = pd.DataFrame()
        img = dataset[image,:,:]
        for DISTANCE in range(1,NUMDIST,DISTSTEP):
            for ANGLE in range(0,NUMANGLES,1):
                GLCM = greycomatrix(img,[DISTANCE],[ANGLE*ANGLESTEP])
                GLCM_Energy = greycoprops(GLCM,'energy')[0]
                df['Energy'] = GLCM_Energy
                GLCM_diss = greycoprops(GLCM,'dissimilarity')[0]
                df['Diss_sim'] = GLCM_diss
                GLCM_hom = greycoprops(GLCM,'homogeneity')[0]
                df['Homogen'] = GLCM_hom
                GLCM_contr = greycoprops(GLCM,'contrast')[0]
                df['contrast'] = GLCM_contr
        image_dataset = image_dataset.append(df)
    return image_dataset


train_images=[]
train_labels=[]
print('Reading training images ....')
for directory_path in glob.glob(path + "train/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path,imagetype)):
        #print(img_path)
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(SIZE,SIZE))
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images=[]
test_labels=[]
print('Reading testing images ....')
for directory_path in glob.glob(path+"test/*"):
    tumor_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path,imagetype)):
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(SIZE,SIZE))
        test_images.append(img)
        test_labels.append(tumor_label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
numTestImages= test_images.shape[0]

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
NUMCLASS = np.max(train_labels_encoded)+1
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded    

print('Extracting features from  ' + str(x_train.shape[0]) + '  training images ....')
if FEATUREEXT==1:
    image_features, kps_train = SIFT_extractor(x_train)
    x_train = np.repeat(x_train,kps_train,axis=0)
    y_train = np.repeat(y_train,kps_train)
    train_labels_encoded = np.repeat(train_labels_encoded,kps_train)
    train_labels = np.repeat(train_labels,kps_train)
        
if FEATUREEXT==2:
    image_features = feature_extractor(x_train)

numFeatures = image_features.shape[0]
sizeFeature = image_features.shape[1]
print('Extracted ' + str(numFeatures) + ' x ' + str(sizeFeature) +' = ' +  str(numFeatures*sizeFeature) + ' featues from training images ....')


X_for_ML = image_features

image_features = np.expand_dims(image_features,axis=0)
X_for_ML = np.reshape(image_features,(x_train.shape[0],-1))
        
print('Applyting classifier on features extracted from training images ....')
RF_model = RandomForestClassifier(n_estimators=50,random_state=42)
RF_model.fit(X_for_ML,y_train)

SVM_model = svm.SVC(decision_function_shape='ovo')
SVM_model.fit(X_for_ML,y_train)


d_train = lgb.Dataset(X_for_ML, label = y_train)
lgbm_params = {'learning_rate':0.05,'boosting_type':'dart',
               'objective':'multiclass','num_leaves':100,
               'max_depth':10,'num_class':NUMCLASS}
lgb_model = lgb.train(lgbm_params,d_train,100)

if modeln ==1:
    modelx = RF_model
if modeln ==2:
    modelx = SVM_model
if modeln == 3:
    modelx = lgb_model

#test_features = feature_extractor(x_test)

if FEATUREEXT==1:
    test_features, kps_test = SIFT_extractor(x_test)
    x_test = np.repeat(x_test,kps_test,axis=0)
    y_test = np.repeat(y_test,kps_test)
    test_labels_encoded = np.repeat(test_labels_encoded,kps_test)
    test_labels = np.repeat(test_labels,kps_test)

if FEATUREEXT==2:
    test_features = feature_extractor(x_test)
    

test_features = np.expand_dims(test_features, axis = 0)
test_for_RF = np.reshape(test_features,(x_test.shape[0],-1))

test_prediction = modelx.predict(test_for_RF)
if modeln ==3:
    test_prediction = np.argmax(test_prediction,  axis = 1)
test_prediction = le.inverse_transform(test_prediction)

print("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

cm = confusion_matrix(test_labels,test_prediction)

fig, ax = plt.subplots(figsize=(8,8))
sns.set(font_scale=2)
sns.heatmap(cm,annot=True,linewidth=0.5,ax=ax)

n = random.randint(0,x_test.shape[0]-1)
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img,axis=0)
input_img_features = feature_extractor(input_img)
input_img_features = np.expand_dims(input_img_features,axis=0)
input_img_for_RF = np.reshape(input_img_features,(input_img.shape[0],-1))

img_prediction = modelx.predict(input_img_for_RF)
if modeln == 3:
    img_prediction = np.argmax(img_prediction, axis=1)
img_prediction = le.inverse_transform([img_prediction])

m1 = round(np.sqrt(len(y_test)))
m2 = m1+1

plt.figure(figsize=(10,10))
for i in range(0,len(y_test)):
    img = x_test[i]
    plt.subplot(m1,m2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    input_img = np.expand_dims(img,axis=0)
    input_img_features = feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features,axis=0)
    input_img_for_RF = np.reshape(input_img_features,(input_img.shape[0],-1))

    img_prediction = modelx.predict(input_img_for_RF)
    if modeln ==3:
        img_prediction = np.argmax(img_prediction, axis=1)
    img_prediction = le.inverse_transform([img_prediction])
    print("The prediction for this image is: ", img_prediction[0])
    print("The actual lable for this image is: ", test_labels[i])
    plt.title("Actual:"+ str(test_labels[i]) + "\n Prediction:" + str(img_prediction[0]), fontsize=10)
plt.show()

if len(y_test)<11:
    num_imgs = len(y_test)
if len(y_test)>=11:
    num_imgs = 11
plt.figure(figsize=(20,20))
randomlist = random.sample(range(0,numTestImages-1), num_imgs)
for i in range(1,num_imgs):
    n = randomlist[i]
    img = x_test[n]
    plt.subplot(1,num_imgs,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    input_img = np.expand_dims(img,axis=0)
    input_img_features = feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features,axis=0)
    input_img_for_RF = np.reshape(input_img_features,(input_img.shape[0],-1))

    img_prediction = modelx.predict(input_img_for_RF)
    if modeln ==3:
        img_prediction = np.argmax(img_prediction, axis=1)
    img_prediction = le.inverse_transform([img_prediction])
    print("The prediction for this image is: ", img_prediction)
    print("The actual lable for this image is: ", test_labels[n])
    plt.title("Prediction:"+str(img_prediction[0]) +"\n Actual:"+ str( test_labels[n]), fontsize=10)
plt.show()


print("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

































