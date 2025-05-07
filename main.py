import os
import csv
import pickle
import logging
import traceback

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from HardNet import HardNet
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import gen_batches
from sklearn import preprocessing as pre
from sklearn.neighbors import LocalOutlierFactor
from keras.applications.resnet import ResNet50

import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

## Visualise the feature space
def fsVisualise(metrics, idData):
    
    # Reduce the dimensionality of metrics for t-SNE transformation
    if metrics.shape[0] > 50 and metrics.shape[1] > 50:
        pca_red = PCA(n_components = 50)
        metrics = pca_red.fit_transform(metrics)
    
    # Perform the t-SNE and feature space visualisation
    if metrics.shape[0] < 30:
        # Set perplexity as a half value to the number of samples for small datasets
        perp = int(metrics.shape[0]/2)
    else:
        perp = 30
    
    tsne_metrics = TSNE(n_components=2, perplexity=perp, n_iter=1000, learning_rate=100, init='pca').fit_transform(metrics)

    # Perform the PCA and feature space visualisation
    pca = PCA(n_components=2)
    pca_metrics = pca.fit_transform(metrics)

    # Visualise the data
    fig, axarr = plt.subplots(2)
    fig.set_size_inches(8, 8)

    tempTitle = "FS visualisations using the " + idData[0] + ' spectrogram. Sensor ' + idData[1] + ', axis ' + idData[2] + '.'
    fig.suptitle(tempTitle, fontsize=12)

    axarr[0].scatter(tsne_metrics[:, 0], tsne_metrics[:, 1], s = 4)
    axarr[0].set(xlabel = "t-SNE 1", ylabel = "t-SNE 2")
    axarr[0].set_title("t-SNE Feature space visualisation")
    #axarr[0].legend(["OK", "NOK"], loc='upper right')
    
    axarr[1].scatter(pca_metrics[:, 0], pca_metrics[:, 1], s = 4)
    axarr[1].set(xlabel = "PCA 1", ylabel = "PCA 2")
    axarr[1].set_title("PCA Feature space visualisation")
    #axarr[1].legend(["OK", "NOK"], loc='upper right')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    visPath = os.path.join('./AD_Spectrograms/data/', idData[0])

    ## Prepare the directory to store the predictions
    if not os.path.exists(visPath):
        os.makedirs(visPath)
    
    fig.savefig(os.path.join(visPath, idData[1] + '_' + idData[2]  + '_FeatureSpace.png'))


# Set the root folder for reading the data
#ROOTDIR = '../../Datasets/Spectrograms_Smeral'
ROOTDIR = '../../Datasets/Spectrograms_MaxMill'

# Set the flag for feature extractor selection
EXTRACTOR = 'ResNet50'  # 'HardNet' or 'ResNet50'

# Set the path for saved images
SAVEPATH = os.path.join('./AD_SpectrogramResults/', EXTRACTOR, 'MaxMill')

# Set the flag for loading the pics
LOAD_IMAGES = True

# Set the flag for visualising the feature space
FS_VISUALISE = False

def main():

    # Set the dict to save the spectrograms
    dictSpectrograms = {}

    # Initialize the sensor list
    sensors = []

    ## Image loader

    # Loop through the spectrograms, sensors, images and runs to get number of images for each run
    for spectName in natural_sort(os.listdir(ROOTDIR)):
        folderPath = os.path.join(ROOTDIR, spectName)
        if os.path.isdir(folderPath):

            for sensorName in natural_sort(os.listdir(folderPath)):
                sensorPath = os.path.join(folderPath, sensorName)
                if os.path.isdir(sensorPath):

                    runCounter = 0
                    for runName in natural_sort(os.listdir(sensorPath)):
                        runPath = os.path.join(sensorPath, runName)
                        if os.path.isdir(runPath):

                            imageCount = 0
                            for image in natural_sort(os.listdir(runPath)):
                                if (image.endswith(".png")):
                                    imageCount += 1
                        runCounter += 1
                sensors.append(sensorName)
        break

    axes = [[] for _ in range(imageCount)]

    # Load the image data
    if LOAD_IMAGES:

        # Loop through the spectrograms
        for spectName in natural_sort(os.listdir(ROOTDIR)):

            # Build the path to the spectrogram folder and update the dict
            folderPath = os.path.join(ROOTDIR, spectName)
            if os.path.isdir(folderPath):
                dictSpectrograms.update({spectName: {}})

                # Loop through the sensors
                for sensorName in sensors:

                    axes = [[] for _ in range(imageCount)]

                    # Build the path to the sensor and update the dict
                    sensorPath = os.path.join(folderPath, sensorName)
                    if os.path.isdir(sensorPath):
                        dictSpectrograms[spectName].update({sensorName: {}})

                        # Loop through the runs to get the images
                        for runName in natural_sort(os.listdir(sensorPath)):

                            # Build the path to the run and set the axis counter to zero
                            runPath = os.path.join(sensorPath, runName)
                            if os.path.isdir(runPath):
                                axisCounter = 0

                                # Load the images
                                for image in natural_sort(os.listdir(runPath)):

                                    # Build the path to the image
                                    imagePath = os.path.join(runPath, image)

                                    # Process only PNG files
                                    if (image.endswith(".png")):

                                        # Convert the image to the numpy array
                                        imageFrame = Image.open(imagePath)
                                        imageMatrix = np.divide(np.array(imageFrame), 255)

                                        axes[axisCounter].append(imageMatrix)
                                        axisCounter += 1

                        # Update the dictionary with the runs for one spectrogram and sensor
                        axisCounter = 0

                        for axis in axes:
                            dictSpectrograms[spectName][sensorName].update({'data' + str(axisCounter): axis})
                            axisCounter += 1

        # Save the spectrogram dictionary
        with open('./spectrograms.pickle', 'wb') as handle:
            pickle.dump(dictSpectrograms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load spectrogram dictionary from pickle
        with open('./spectrograms.pickle', 'rb') as handle:
            dictSpectrograms = pickle.load(handle)


    ## Run the images through HardNet and ResNet-50

    # Prepare the feature extractors and classifiers
    hardNet = HardNet()
    resNet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    clf = LocalOutlierFactor(n_neighbors=4)

    anomalyScores = []
    anomalyScoresMed = []

    # Loop through the spectrograms, sensors and sensor data
    for spectrograms, _ in dictSpectrograms.items():

        avgAnomalyScore = np.zeros(runCounter)

        for sensors, _ in dictSpectrograms[spectrograms].items():

            sensorData = list(dictSpectrograms[spectrograms][sensors].keys())

            for data in sensorData:

                preprocessedData = []

                # Convert the images to the 32x32 size
                images = dictSpectrograms[spectrograms][sensors][data]
                
                # HardNet
                if EXTRACTOR == 'HardNet':
                    for img in images:
                        #preprocessedData.append(cv.resize(cv.cvtColor(np.squeeze(img), cv.COLOR_GRAY2RGB), (32, 32), interpolation=cv.INTER_AREA))
                        preprocessedData.append(cv.resize(np.squeeze(img), (32, 32), interpolation = cv.INTER_AREA))
                    
                    preprocessedData = np.array(preprocessedData)
                    preprocessedData = np.expand_dims(preprocessedData, axis=3)

                # ResNet50
                if EXTRACTOR == 'ResNet50':
                    
                    for img in images:
                        # Convert the image to uint8 and ensure it has three channels
                        img_8u = cv.convertScaleAbs(np.squeeze(img))
                        img_rgb = cv.cvtColor(img_8u, cv.COLOR_GRAY2RGB)
                        resized_img = cv.resize(img_rgb, (32, 32), interpolation=cv.INTER_AREA)
                        preprocessedData.append(resized_img)

                    preprocessedData = np.array(preprocessedData)

                # Run the images through the feature extractor

                # Generate batches
                batches = gen_batches(preprocessedData.shape[0], 20)
                fsRun = True
                
                if EXTRACTOR == 'HardNet':
                    # Get HardNet features  
                    for batch in batches:
                        temp = hardNet.forward(preprocessedData[batch])
                        
                        if fsRun:
                            metrics = temp
                            fsRun = False
                        else:
                            metrics = np.concatenate((metrics, temp), axis=0)

                if EXTRACTOR == 'ResNet50':
                    # Get ResNet50 features  
                    for batch in batches:
                        temp = batch
                        temp = resNet50.predict(preprocessedData[batch])
                        temp = temp.reshape(temp.shape[0], -1)
                        
                        if fsRun:
                            metrics = temp
                            fsRun = False
                        else:
                            metrics = np.concatenate((metrics, temp), axis=0)
                
                # Fit the classifier and get the average anomaly score from sensors and runs
                clf.fit_predict(metrics)

                anomalyScore = clf.negative_outlier_factor_
                avgAnomalyScore = anomalyScore + avgAnomalyScore

                # Visualise the feature space
                if FS_VISUALISE:
                    fsVisualise(metrics, [spectrograms, sensors, data])

                
        # Get the average anomaly score, save it raw and with median substraction
        avgAnomalyScore = avgAnomalyScore / (imageCount * len(sensors))

        anomalyScores.append(avgAnomalyScore)
        anomalyScoresMed.append(avgAnomalyScore - np.median(avgAnomalyScore))
        
        output_dir = os.path.join(SAVEPATH, spectrograms)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.bar(np.arange(0, runCounter), avgAnomalyScore)
        plt.savefig(os.path.join(os.path.join(SAVEPATH, spectrograms), 'AnomalyScore.png'))
        plt.close()

        with open(os.path.join(os.path.join(SAVEPATH, spectrograms), 'AnomalyScore.txt'), "w") as output:
            output.write(str(avgAnomalyScore))
    
    # Get std deviations of anomaly scores from spectrograms and runs
    anomalyScoresNP = np.array(anomalyScores)
    spectDev = []
    runDev = []

    for spectrogramScore in anomalyScoresNP:
        spectDev.append(np.std(spectrogramScore))

    for runScore in np.transpose(anomalyScoresNP):
        runDev.append(np.std(runScore))

    # Save the anomaly scores to CSV
    with open(os.path.join(SAVEPATH, 'anomalyScores.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(anomalyScores)

    if False:
        # Save the spectrogram scores to CSV
        with open(os.path.join(SAVEPATH, 'spectrogramScores.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(spectDev)

        # Save the run scores to CSV
        with open(os.path.join(SAVEPATH, 'runScores.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(runDev)

    print('Evaluation finished!')

if __name__ == '__main__':
    main()
