from glob import glob
import numpy as np
import cv2

class DataProcessing(object):
    
    def __init__(self, dataFolderPath):
        self.dataFolderPath = dataFolderPath
        
    def getData(self):
        #10304 since all images are (112, 92), therefore vector is 10304-dimensional
        X = np.zeros((10304, 0))
        y = []
        
        #populate X featurre matrix and y response vector
        folderPaths = glob(self.dataFolderPath)
        for folderPath in folderPaths:
            imagePaths = glob(folderPath + '/*')
            for imagePath in imagePaths:
                personID = imagePath.split('/')[1]
                y.append(personID)
                
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                self.shape = image.shape
                
                imageVector = image.ravel().reshape((-1, 1))
                
                X = np.hstack((X, imageVector))
                
        y = np.array(y)
        
        return X, y

class Eigenface(object):
    
    def __init__(self, K, X):
        
        #calculate mean face
        self.meanFace = X.mean(1).reshape((-1, 1))

        #calculate mean-normalized matrix
        A = X - np.tile(self.meanFace, (1, X.shape[1]))
    
        #calculate eigen decomposition for A^TA instead of AA^T (speed)
        M = np.matmul(A.T, A)
        eigenValues,eigenVectors = np.linalg.eig(M)
        
        #sort the eigenvectors and eigenvalues
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        #choose principal components
        eigenVectors = eigenVectors[:, 0:K]
        
        #calculate projection matrix u
        self.u = np.matmul(A, eigenVectors)
        
        #calculate eigenface subspace
        self.eigenfaceSubspace = np.matmul(self.u.T, A)
        
    def predict(self, testSample, y):
        #project new mean-normalized test sample onto eigenface subspace
        testSample = testSample.reshape((-1, 1)) - self.meanFace
        testProjected = np.matmul(self.u.T, testSample)
    
        #classify as closest in euclidean distance
        predictedPerson = y[np.linalg.norm(self.eigenfaceSubspace - np.tile(testProjected, (1, self.eigenfaceSubspace.shape[1])), axis=0).argmin()]
        
        return predictedPerson
        

def main():
    dataProcessor = DataProcessing('data/*')
    
    X, y = dataProcessor.getData()
    
    K = 100
    eigenface = Eigenface(K, X)
    
    for i in range(X.shape[1]):    
        testSample = X[:, i]
        print testSample.shape
        predictedPerson = eigenface.predict(testSample, y)
        print predictedPerson, y[i]
        
if __name__ == "__main__":
    main()
