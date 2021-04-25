def readTrainTestImages(trainpath,testpath):
	import glob
	import pandas as pd
	import numpy as np
    from tqdm import tqdm
    import matplotlib.image as mpimg
    trainImages = []
    testImages = []
    trainimgfiles = glob.glob(trainpath + '/*.png')
    testimgfiles = glob.glob(testpath + '/*.png')
    trainimgfiles = sorted(trainimgfiles, key=lambda x: int(x.split("\\")[1].split(".")[0]))
    testimgfiles = sorted(testimgfiles, key=lambda x: int(x.split("\\")[1].split(".")[0]))
    for i in tqdm(range(len(trainimgfiles))):
        # Read Images
        img = mpimg.imread(trainimgfiles[i])   
        trainImages.append(img)        
    print('read ' + str(i) + " training images")   
    
    for i in tqdm(range(len(testimgfiles))):
        # Read Images
        img = mpimg.imread(testimgfiles[i])   
        testImages.append(img)        
    print(f'read {i} test images')
            
    
    trainImages = np.array(trainImages, dtype='float32')
    testImages = np.array(testImages, dtype='float32')
    
    trainlabels = pd.read_csv(trainpath + '/trainlabels.csv',index_col=False)
    testlabels = pd.read_csv(testpath + '/testlabels.csv',index_col=False)
    trainlabels.drop(trainlabels.columns[0], axis=1, inplace=True)
    testlabels.drop(testlabels.columns[0], axis=1, inplace=True)
    
    return trainImages, trainlabels, testImages, testlabels

trainpath = './train-pngs'
testpath = './test-pngs'
trainData, trainLabels, testData, testLabels = readTrainTestImages(trainpath,testpath)