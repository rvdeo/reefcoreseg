#ReefCoreSeg framework

#Created by Ratneel Deo on 18 August 2022

#Data Requirements - Reef core image data and Multi scan core logger data ()



from tkinter.tix import IMAGETEXT
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import itertools
import shutil


from numpy import unique
from numpy import where
import scipy.stats
from scipy import linalg
from skimage import data
from skimage.color import rgb2gray
    

from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.utils import shuffle
from sklearn.neighbors import NearestCentroid


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import os
import time
import math
import warnings

from sklearn.preprocessing import MinMaxScaler



from numpy import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.model_selection import train_test_split 
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
import matplotlib.patches as mpatches

from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings("ignore")




print("Starting Reef Core Segmentation Engine")
def gmm_bic_analysis(path,X,range_n_clusters):
    
    folder_name = path+"/gmm_results"
    os.mkdir(folder_name)
    

    lowest_bic = np.infty
    bic = []
    n_components_range = range_n_clusters
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    '''
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {best_gmm.covariance_type} model, "
        f"{best_gmm.n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    '''
    title = folder_name+"/gmm_bic.png"
    plt.savefig(title)
    #plt.show()
    return folder_name+"/"


def data_analysis_exp_325(path):
    url = "http://publications.iodp.org/proceedings/325/101/101_t1.htm#wp1016591"

    table = pd.read_html(url, header=1)[0]
    table = table[1:]

    cols=table.columns[3:]
    table[cols] = pd.to_numeric(table[cols].stack(), errors='coerce').unstack()
    stats = table[["Hole","PenetrationdepthDSF-A (m)", "Drill string"]]

    stats.set_index("Hole", inplace=True)
    plt.figure(figsize=(10,10))
    stats.plot.bar(stacked=True,  )
    plt.legend(["Length of Core","Drilling Depth"])
    plt.ylabel("Depth in Meters")
    plt.xlabel("Drill Hole")
    plt.savefig(path+"Drill_core_description.png")
    #plt.show()

def kmeans_shilouette_analysis(path,X,range_n_clusters):
    
    folder_name = path+"/kmeans_results"
    os.mkdir(folder_name)

    distorsions = []

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        #store teh distrotions for analysis in elbo plot
        distorsions.append(clusterer.inertia_)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        title = folder_name+"/shil_"+str(n_clusters)+".png"
        plt.savefig(title)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range_n_clusters, distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.savefig(folder_name+"/elbow_plot.png")

    #plt.show()
    return folder_name+"/"


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


def create_res_folder():
    
    folder_name = 'results_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(folder_name)
    return folder_name+'/'


def read_image(scale_percent):
    # read the image
    image = cv.imread("/home/ratneel/Dropbox/RatneelPhd/Experiments/Clustering_Segmentation/images/img.jpg")
    print('Original Dimensions : ',image.shape)
    
    # resize image
    # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    print('Resized Dimensions : ',resized_image.shape)

    # convert to RGB
    resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

    #cv.imshow("Resized image", resized_image)
    cv.imwrite("Resized_image.png", resized_image)
    #print('Resized Dimensions : ',resized_image)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = resized_image.reshape((-1, 3))
    
    # convert to float
    pixel_values = np.float32(pixel_values)
    
    return pixel_values,resized_image

def read_physical_prop():
    rows=[*range(0, 17, 1)]
    #print(rows)
    df = pd.read_table("/home/ratneel/Dropbox/RatneelPhd/Experiments/Clustering_Segmentation/data/325-M0033A_phys_prop.tab", skiprows=rows, delimiter='\t',header=1)
    #df = df.fillna(0)
    #print(df.columns)

    all_prop = df[df.Label.str.contains("16R")==True]
    phy_prop = all_prop[['WBD [g/cm**3]','Poros [% vol]', 'Resist electr [Ohm m]']]
    phy_prop['Poros [% vol]'] = phy_prop['Poros [% vol]']/100.0
    phy_prop.reset_index(inplace=True)
    return phy_prop.fillna(0)



def add_labels():
    lb_df = pd.read_table("/home/ratneel/Dropbox/RatneelPhd/Experiments/Clustering_Segmentation/data/Unbalanced_325.csv", delimiter=',',header=0)
    lb_df.reset_index(inplace=True)
    phy_prop = read_physical_prop()

    phy_prop["Species"] = phy_prop['WBD [g/cm**3]']==True#np.where(lb_df["Bulk"] == phy_prop['WBD [g/cm**3]'], lb_df["Species"], '') #create a new column in df1 to check if prices match

    for item in range(len(phy_prop['WBD [g/cm**3]'])):
        for value in range(len(lb_df["Bulk"])):
            if  phy_prop['WBD [g/cm**3]'][item] == lb_df["Bulk"][value]: #and phy_prop['Resist electr [Ohm m]'][item] == round(lb_df["Resistivity"][value],3) :
                if  phy_prop['Resist electr [Ohm m]'][item] == round(lb_df["Resistivity"][value],3):
                    phy_prop["Species"][item] = lb_df["Species"][value]
                    #print(item, ' - ' , value , ' -- ', phy_prop['WBD [g/cm**3]'][item] , ' -- ', ' ** ', phy_prop['Resist electr [Ohm m]'][item] , ' -- ', lb_df["Resistivity"][value], ' ** ' , phy_prop['Species'][item] )

    phy_prop.to_csv('data/phyprop.csv', header = True) 
    
    return phy_prop




def kmeans_seg(path, X , range_n_clusters, img):
    scores = []
    with open(path+'_kmeans_results.txt', 'a') as f:
        f.write("Culsters  \tSilhouette_score : \tCalinski score :  \tDavies score : \n")
        c_labels = []
        for n_clusters in range_n_clusters:
        
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            cluester_centers = clusterer.fit(X).cluster_centers_

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            calinski_score = calinski_harabasz_score(X, cluster_labels)

            davies_score = davies_bouldin_score(X, cluster_labels)

            #results =  "For n_clusters = "+ str(n_clusters) + " \tSilhouette_score : " + str(silhouette_avg) +  " \tCalinski score : " + str(calinski_score) + " \tDavies score : " + str(davies_score) +'\n'
            results =  " & "+ str(n_clusters) + " & " + str(silhouette_avg) +  " & " + str(calinski_score) + " & " + str(davies_score) +'\\\\\n'
            
            f.write(results)
            scores.append([silhouette_avg,calinski_score,davies_score])
            print(results)

            draw_clusters(img, cluster_labels, n_clusters,path+"kmeans")
            c_labels.append(cluster_labels)
    return scores , c_labels


def gmm_seg(path, X , range_n_clusters,img):
    scores = []
    with open(path+'_gmm_results.txt', 'a') as f:
        f.write("Culsters  \tSilhouette_score : \tCalinski score :  \tDavies score : \n") 
        c_labels = []
        for n_clusters in range_n_clusters:
        
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            cluester_centers  = np.empty(shape=(clusterer.n_components, X.shape[1]))
            for i in range(clusterer.n_components):
                density = scipy.stats.multivariate_normal(cov=clusterer.covariances_[i], mean=clusterer.means_[i]).logpdf(X)
                cluester_centers[i, :] = X[np.argmax(density)]

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            calinski_score = calinski_harabasz_score(X, cluster_labels)

            davies_score = davies_bouldin_score(X, cluster_labels)

            #results =  "For n_clusters = "+ str(n_clusters) + " \tSilhouette_score : " + str(silhouette_avg) +  " \tCalinski score : " + str(calinski_score) + " \tDavies score : " + str(davies_score) +'\n'
            results =  " & "+ str(n_clusters) + " & " + str(silhouette_avg) +  " & " + str(calinski_score) + " & " + str(davies_score) +'\\\\\n'
            
            f.write(results)
            scores.append([silhouette_avg,calinski_score,davies_score])
            print(results)

            draw_clusters(img, cluster_labels, n_clusters,path+"gmm")
            c_labels.append(cluster_labels)

    return scores , c_labels

def aglo_seg(path, X, range_n_clusters, img):

    # Agglomerative clustering
    
    scores = []
    with open(path+'_aglometric_results.txt', 'a') as f:
        f.write("Culsters  \tSilhouette_score : \tCalinski score :  \tDavies score : \n") 
        '''
            plt.figure(figsize=(10, 5))  
            plt.title("Dendrograms")  
            dend = shc.dendrogram(shc.linkage(X, method='ward'))
            plt.savefig(path+"dendogram.png")
        '''
        
        c_labels = []

        for n_clusters in range_n_clusters:
            # define the model
            clusterer = AgglomerativeClustering(n_clusters)
            # fit model and predict clusters
            yhat = clusterer.fit(X)
            yhat_2 = clusterer.fit_predict(X)

            clf = NearestCentroid()
            clf.fit(X, yhat_2)
            print("Centroids:")
            print(clf.centroids_)

            cluster_labels = yhat_2
            cluester_centers = clf.centroids_

            # retrieve unique clusters
            clusters = unique(yhat)

            # Calculate cluster validation metrics
            score_AGclustering_s = silhouette_score(X, yhat.labels_, metric='euclidean')
            score_AGclustering_c = calinski_harabasz_score(X, yhat.labels_)
            score_AGclustering_d = davies_bouldin_score(X, yhat_2)

            print('Silhouette Score: %.4f' % score_AGclustering_s)
            print('Calinski Harabasz Score: %.4f' % score_AGclustering_c)
            print('Davies Bouldin Score: %.4f' % score_AGclustering_d)

            #results =  "For n_clusters = "+ str(n_clusters) + " \tSilhouette_score : " + str(score_AGclustering_s) +  " \tCalinski score : " + str(score_AGclustering_c) + " \tDavies score : " + str(score_AGclustering_d) +'\n'
            results =  " & "+ str(n_clusters) + " & " + str(score_AGclustering_s) +  " & " + str(score_AGclustering_c) + " & " + str(score_AGclustering_d) +'\\\\\n'
            
            f.write(results)
            scores.append([score_AGclustering_s,score_AGclustering_c,score_AGclustering_d])
            print(results)

            draw_clusters(img, cluster_labels, n_clusters,path+"aglo")
            c_labels.append(cluster_labels)
    return scores , c_labels



def dbscan_seg(path, X , range_n_clusters,img):
    scores = []
    with open(path+'_dbscan_results.txt', 'a') as f:
        
        f.write("Culsters  \tSilhouette_score : \tCalinski score :  \tDavies score : \n" )
        
        c_labels = []

        epsilon = 0.2
        for n_clusters in range_n_clusters:
            epsilon = n_clusters
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = DBSCAN(eps=epsilon, min_samples=2)#(n_components=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
           

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = round(silhouette_score(X, cluster_labels),3)
            calinski_score = round(calinski_harabasz_score(X, cluster_labels),3)
            davies_score = round(davies_bouldin_score(X, cluster_labels),3)

            #results =  "For n_clusters = "+ str(n_clusters) + " \tSilhouette_score : " + str(silhouette_avg) +  " \tCalinski score : " + str(calinski_score) + " \tDavies score : " + str(davies_score) +'\n'
            results =  " & "+ str(epsilon) + " & " + str(silhouette_avg) +  " & " + str(calinski_score) + " & " + str(davies_score) +'\\\\\n'
            
            epsilon = epsilon * 2

            f.write(results)
            scores.append([silhouette_avg,calinski_score,davies_score])
            print(results)

            cluester_centers = ""

            draw_clusters(img, cluster_labels, n_clusters,path+"_dbscan")
            c_labels.append(cluster_labels)
    return scores , c_labels
            
   






    
def draw_clusters(img, labels, clusters,path):
    
    fig = plt.figure(figsize=(2,17))
    ax = fig.add_subplot(111)
    if "aglo" in path:
        cm = 'jet'
    elif "kmeans" in path:
        cm = 'ocean'
    else:
        cm = "terrain"
    
    out_image = (np.reshape(labels, [img.shape[0], img.shape[1]]))

    colmap = plt.get_cmap(cm, np.max(out_image)-np.min(out_image)+1)

   
    pc = ax.imshow(out_image, cmap=colmap)
    ax.set_aspect('auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="15%", pad=0.05)

    plt.colorbar(pc, cax=cax)

    name = path+"_"+str(clusters)+".png" 
    plt.savefig(name, bbox_inches='tight')


    print("image drawn ", img.shape)




def Cluster_Module(pth, clusters, algorithms,scale_per, stack_data, image_only):

    

    # Read data
    scale_percent = scale_per 
    Y,img = read_image(scale_percent)

    
    #visualise drill data
    #data_analysis_exp_325(path)

    #choose clusters
    range_n_clusters = clusters#  [ 6]#, 4, 5, 6, 7, 8, 9, 10]

    algo = algorithms # "0"
        


    if stack_data == True:  
        path = pth+ 'stacked/' 
        os.mkdir(path)
        dat = add_labels()
        #print(dat["Species"].unique())
        

        phy_dat = dat[["WBD [g/cm**3]","Poros [% vol]","Resist electr [Ohm m]"]].to_numpy()
        print(phy_dat.shape)
         
        comb_data = np.zeros([img.shape[0],img.shape[1]+1,img.shape[2]])
        print(comb_data.shape)

        

        num_phy_prop = phy_dat.shape[0]

        expand_ratio = math.ceil(comb_data.shape[0]/num_phy_prop)

        phy_prop_index = 0

        for i in range(comb_data.shape[0]):
            
            comb_data[i][:-1] = img[i][:]
            comb_data[i][-1:] = phy_dat[phy_prop_index]

            if i != 0 and i % expand_ratio == 0 :
                phy_prop_index += 1

        #np.savetxt(path+"combined_data.csv", comb_data, delimiter=",", header='x, y, z', fmt ='%s',comments='')
        #pd.DataFrame(comb_data).to_csv("path/to/file.csv", header=None, index=None)
        #print(comb_data)
        # convert it to stacked format using Pandas

        
        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        #X = img.reshape((-1, 3))
        #img = comb_data
        X = comb_data.reshape((-1, 3))
    
        # convert to float
        X = np.float32(X)

        if "0" in algo:
            folder = kmeans_shilouette_analysis(path,X,range_n_clusters)
            score_kmeans, clabels = kmeans_seg(folder, X , range_n_clusters, comb_data)
            print (score_kmeans)

        if "1" in algo:
            folder=gmm_bic_analysis(path,X,range_n_clusters)
            score_gmm , clabels = gmm_seg(folder, X , range_n_clusters,comb_data)
            print(score_gmm)

        if "2" in algo:
            score_dbscan , clabels = dbscan_seg(path, X , range_n_clusters,comb_data)
            print(score_dbscan)

        if "3" in algo:
            score_aglo , clabels = aglo_seg(path, X, range_n_clusters, comb_data )
            print(score_aglo)

    if image_only == True:
        path = pth+  "image_only/"
        os.mkdir(path)
        
        X = img.reshape((-1, 3))
        comb_data = img

        if "0" in algo:
            folder = kmeans_shilouette_analysis(path,X,range_n_clusters)
            score_kmeans, clabels  = kmeans_seg(folder, X , range_n_clusters, comb_data)
            print (score_kmeans)

        if "1" in algo:
            folder=gmm_bic_analysis(path,X,range_n_clusters)
            score_gmm , clabels = gmm_seg(folder, X , range_n_clusters,comb_data)
            print(score_gmm)

        if "2" in algo:
            score_dbscan , clabels = dbscan_seg(path, X , range_n_clusters,comb_data)
            print(score_dbscan)

        if "3" in algo:
            score_aglo, clabels  = aglo_seg(path, X, range_n_clusters, comb_data )
            print(score_aglo)

    

   
    return clabels, comb_data
    





################################################################################################################################
################################################################################################################################
################################################################################################################################


# Classicication Module


def read_preprocess_data():
    
    df = pd.read_csv("/home/ratneel/Dropbox/RatneelPhd/Experiments/Clustering_Segmentation/data/Unbalanced_325.csv")
 

    df_filtered = df[["Bulk","Porosity", "Resistivity"]]
    print(df_filtered.describe())

    test_df = pd.read_csv("/home/ratneel/Dropbox/RatneelPhd/Experiments/Clustering_Segmentation/data/phyprop.csv")
   

    test_df_filtered = test_df[['WBD [g/cm**3]','Poros [% vol]', 'Resist electr [Ohm m]']]

    normalize = True

    if normalize:


        #Use minmax normalization to get data in 0-1 rnage for lstm model
        x = df_filtered.values #returns a numpy array of the data frame
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        
        
        # Splitting data in traning and testting set
        all_data = pd.DataFrame(x_scaled).values

        test_data = min_max_scaler.fit_transform(test_df_filtered.values)

    else:
        all_data = df_filtered.values
        test_data = test_df_filtered.values


    encoded_labels = df[["Species"]]
    
    le = preprocessing.LabelEncoder()
    
    encoded_labels =  le.fit_transform(encoded_labels)
    labels = list(le.classes_)


    balance_data = False
    if balance_data:
        resample = NeighbourhoodCleaningRule()
        all_data, encoded_labels = resample.fit_resample(all_data, encoded_labels)

    
    desc_data = True
    if desc_data:

        counter = Counter(encoded_labels)

        for k,v in counter.items():
            per = v / len(encoded_labels) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

        #plt.bar(counter.keys(), counter.values())
        plt.bar(labels, counter.values())
        plt.xticks(rotation = 30)
        
        #encoded_labels.value_counts().plot(kind='bar')
        plt.show()

    
    x_train, x_test, y_train, y_test = train_test_split(all_data, encoded_labels, test_size=0.34, random_state=1)

    return x_train, x_test, y_train, y_test, labels, test_data, le

 
    
def scipy_models(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num, test_data, labels,pth):

    print(run_num, ' is our exp run')

    tree_depth = 10
 
    name = ""
    
    if type_model ==0: 
        name = "gnb"+str(run_num)+".png"
        model = GaussianNB()
    
    elif type_model ==1:
        name = "svc"+str(run_num)+".png"
        model = SVC(kernel = 'linear', C = 1)

    elif type_model ==2:
        model = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=2000, solver='lbfgs',  learning_rate_init=learn_rate, activation = 'logistic' )  
        name = "fnn"+str(run_num)+".png"
       
               
    
    elif type_model ==3:
        model = RandomForestClassifier(n_estimators=100, max_depth=tree_depth, random_state=run_num)
        name = "rfc"+str(run_num)+".png"

    elif type_model ==4:
        name = "knn"+str(run_num)+".png"
        model = KNeighborsClassifier(n_neighbors = 7)

    elif type_model ==5:
        #kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        #kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
        kernel = 1.0 * RBF(1.0)
        model = GaussianProcessClassifier(kernel=kernel, random_state=run_num)
        name = "gpr"+str(run_num)+".png"

    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)


    
    perf_test = accuracy_score(y_pred_test, y_test) 
    perf_train = accuracy_score(y_pred_train, y_train) 
    cm = confusion_matrix(y_pred_test, y_test) 
    #plot_confusion_matrix(model, x_test, y_test) 
    #plt.show()
    
    print(cm, 'is confusion matrix')
    if run_num == 0:
        pc = ConfusionMatrixDisplay.from_estimator( model, x_test, y_test, display_labels=labels, cmap=plt.cm.OrRd, normalize="pred", colorbar=False)
        plt.savefig(pth+"cm_"+name)
        #plt.show()
    
    #metrics.plot_roc_curve(model, x_test, y_test)
    cr = metrics.classification_report(y_test, y_pred_test, target_names=labels)
    
    #RocCurveDisplay.from_predictions(y_test, y_pred_test)
    #plt.show()
    #plt.savefig("roc_"+name)
    #auc = roc_auc_score(y_pred_test, y_test, average='macro', multi_class='ovo')
    #print(auc) 

    #make predictions on testing data -  CORE 33A
    test_data_pred = model.predict(test_data)

    return perf_test, name, cr , test_data_pred#,perf_train



    
def draw_classified_core(predictions, labels, title):

    c = predictions #np.array(predictions)
        
    ig_dat = np.swapaxes(np.tile(c,(5,1)),0,1)

    fig, ax = plt.subplots(1,1)
    
    im = ax.imshow(ig_dat, cmap="inferno")
    
    #plt.set_title(title)
    ax.get_xaxis().set_visible(False)
    #plt.set_aspect('auto')
    values = np.unique(ig_dat.ravel())

    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [ im.cmap(im.norm(value)) for value in values]
    
    #patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
    patches = [ mpatches.Patch(color=colors[i], label= labels[values[i]]) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, bbox_to_anchor=(0.5, 1.2), loc='upper center')
    
    plt.tight_layout()
    plt.savefig(title)
    plt.show()




def Classification_module(pth, model_list): 

    plt.rcParams["figure.figsize"] = (10,10)

    max_expruns = 1

 
    learn_rate = 0.01
    hidden = 32

    prob = 'classifification' #  classification  or regression 
   

    # classifcation accurary is reported for classification and RMSE for regression

    print(prob, ' is our problem')

    

    modList = model_list # [0,1,2,3,4]



    model_perf = []
    clas_report = []
    
    for run_num in range(0,max_expruns): 
        x_train, x_test, y_train, y_test, labels, test_data, label_enc = read_preprocess_data()
          
        names = []
        run_perf = []
        run_rep = []
        test_classifications = []
        for model in modList:

            accr, name, cr , test_classification = scipy_models(x_train, x_test, y_train, y_test, model, hidden, learn_rate, run_num, test_data, labels, pth) #SGD 
            names.append(name[:3])
            run_perf.append(accr)
            run_rep.append(cr)
            test_classifications.append(test_classification)

        model_perf.append(run_perf)
        clas_report.append(run_rep)

    
    le_name_mapping = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    inv_mapping = dict(map(reversed, le_name_mapping.items()))

    if len(modList) > 1:

        plt.rcParams["figure.figsize"] = (7,10)
        fig, ax = plt.subplots(1,len(modList))
        
        
        for model in range(len(modList)):
            print("=="*10)
            print(names[model])
            print("--"*10) 
            print(clas_report[run_num][model])

        
            
            c = np.array(test_classifications[model])
            
            ig_dat = np.swapaxes(np.tile(c,(5,1)),0,1)
            im = ax[model].imshow(ig_dat, cmap="inferno")
            ax[model].set_title(names[model])
            ax[model].get_xaxis().set_visible(False)
            
            values = np.unique(ig_dat.ravel())

            # get the colors of the values, according to the 
            # colormap used by imshow
            colors = [ im.cmap(im.norm(value)) for value in values]
        
            #patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
            patches = [ mpatches.Patch(color=colors[i], label= inv_mapping[values[i]]) for i in range(len(values)) ]
            # put those patched as legend-handles into the legend
            ax[model].legend(handles=patches, bbox_to_anchor=(0.5, 1.2), loc='upper center')

        plt.tight_layout()
        plt.savefig(pth+"classifiation_perfomance.png")
        #plt.show()
    
    else:
        print(test_classifications[0])
        title = pth+names[0]+"classifiation_perfomance.png"

        draw_classified_core(test_classifications[0], inv_mapping, title)

        
     
    
    print("Model ", names)
    print(np.mean(model_perf, axis=0))
    print(np.std(model_perf, axis=0))


    return test_classifications, inv_mapping


    ############################################################################
    ############################################################################


def annotate_with_GP_and_KNN(pth):
    print("annotation image")



plt.rcParams.update({'font.size': 18})   

def main():
    #create output folder
    pth = create_res_folder()

    annotate = True
    if annotate:
        # choose list of classifiers [GaussianNatieveBias:0, SVC:1, RandomForest:2, NeuralNet:3, KnearestNeighbour:4]
        model_list = [0,1,2,3,4]
        test_classifications, labels =  Classification_module(pth, model_list)
        print(test_classifications)
        print(labels)

        #number of clusters
        clusters = [len(np.unique(test_classifications))]

        # clustering algorithms to use [kmeans:0, gmm:1, dbscan:2, aglo:3 ]
        algorithms = "1"

        scale_per = 5
        stack_data = True
        image_only = False
        clabels, img = Cluster_Module(pth, clusters, algorithms,scale_per,stack_data, image_only)

        draw_clusters(img, clabels, clusters,pth)


        ##################################################################################################
        #################################################################################################

        #rescale the predictions for the image
         


    else: 
        # choose list of classifiers [GaussianNatieveBias:0, SVC:1, RandomForest:2, NeuralNet:3, KnearestNeighbour:4]
        model_list = [0,1,2,3,4]
        test_classifications, labels =  Classification_module(pth, model_list)
        print(test_classifications)
        print(labels)



        #number of clusters
        clusters = [3,4,5,6,7,8,9,10]

        # clustering algorithms to use [kmeans:0, gmm:1, dbscan:2, aglo:3 ]
        algorithms = "013"

        scale_per = 5
        stack_data = True
        image_only = True
        
        clabels, img = Cluster_Module(pth, clusters, algorithms,scale_per,stack_data, image_only)
   

    #make a copy of the code file used to generate the results within the results sub folder
    # in the source variable
    source = 'ReefCoreSegAnnotate.py'
    
    # storing the destination path in the
    # destination variable
    destination = pth+source
    
    # calling the shutil.copyfile() method
    shutil.copyfile(source,destination)




if __name__ == '__main__':
     main() 







################################################################################################################################
################################################################################################################################
################################################################################################################################






