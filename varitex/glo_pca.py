import os 
import sys 
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import argparse

def fit_pca_in_dataloader(dataLoader, target_dir, n_pca, latent_dim):
    X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _)
             in zip(tqdm(range(n_pca // dataLoader.batch_size), 'collect data for PCA'), 
                    dataLoader)
        ])
    pca = PCA(n_components=latent_dim)
    pca.fit(X_pca)
    Z = np.empty((len(dataLoader.dataset), latent_dim))
    for X, idx in tqdm(dataLoader, 'pca projection'):
        Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))
    np.save(target_dir, Z)

def fit_pca_in_npy_file(source_file, target_dir, n_pca, latent_dim,batch_size):
    images = np.load(source_file, mmap_mode='r')
    #images_pca = np.vstack([x.reshape(((x).size)) for x in images[:n_pca]])
    print('Fitting PCA')
    #pca = IncrementalPCA(n_components=latent_dim, batch_size=batch_size)
    #pca.fit(images_pca)
    ipca = IncrementalPCA(n_components=latent_dim)
    for idx in tqdm(len(images)):
        X = images[idx*batch_size: (idx+1)*batch_size]
        images_pca = np.vstack([x.reshape(((x).size)) for x in X])
        ipca.partial_fit(images_pca)
    print('Applying PCA')
    Z = np.empty((len(images), latent_dim))
    for idx, X in tqdm(zip(range(len(images)), images), 'pca projection', total=len(images)):
        Z[idx] = pca.transform(X.reshape(1,(X).size))
    np.save(target_dir, Z)
    

#This functions fits a pca to a dataset at location and saves the latentcodes in target_dir
def fit_pca_transform_to_npy(source_file, target_dir, n_pca, latent_dim, dataLoader,batch_size):
    if(dataLoader=='DataLoader'):
        fit_pca_in_dataloader(source_file, target_dir, n_pca, latent_dim)
    elif(dataLoader=='NumpyFile'):
        fit_pca_in_npy_file(source_file, target_dir, n_pca, latent_dim,batch_size)
    else:
        print('Argument dataLoader needs to be either "DataLoader" or "NumpyFile"')


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pca', type=int, help='Number of samples to choose for pca', default=10000)
    parser.add_argument('--source_path',  help='Path of the npy file dataset or dataset for DataLoader',default=None)
    parser.add_argument('--latent_dim', type=int, help='Dimension of latent space', default=256)
    parser.add_argument('--target_path',  help='Path of where to save resulting latent codes',default='pcaLatents.npy')
    parser.add_argument('--batch_size', type=int, help='batch size for incremental pca', default=5000)
    parser.add_argument('--dataLoader', help='One of either "DataLoader" or "NumpyFile", dependent on how you want to load the data', default='NumpyFile')
    args=parser.parse_args()
    print(args.n_pca)
    if(args.source_path==None):
        #source_file = '/home/matthias/ETH/Thesis/Data/FFHQ/preprocessed_dataset/images.npy'
        source_file = '/cluster/work/hilliges/shared/face_datasets/FFHQ/preprocessed_dataset/images.npy'
    fit_pca_transform_to_npy(source_file, args.target_path, args.n_pca, args.latent_dim, args.dataLoader, args.batch_size)
