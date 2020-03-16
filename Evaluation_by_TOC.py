import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from urllib.request import urlopen
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


utils = importr('utils')
# utils.install_packages('TOC')
# utils.install_packages('Rcpp')
# utils.install_packages('codetools')
Rcpp = robjects.packages.importr('Rcpp')
TOC = robjects.packages.importr('TOC')

index = robjects.r('index <- raster(system.file("external/Prob_Map2.rst", package = "TOC"))')
boolean = robjects.r('boolean <- raster(system.file("external/Change_Map2b.rst", package = "TOC"))')
mask = robjects.r('mask <- raster(system.file("external/MASK4.rst", package = "TOC"))')

print(robjects.r('index'))
print(robjects.r('boolean'))
print(robjects.r('mask'))

print(robjects.r('unique(index)'))
print(robjects.r('unique(boolean)'))
print(robjects.r('unique(mask)'))

toc = robjects.r('tocd <- TOC(index, boolean, mask, nthres = 100)')
robjects.r('plot(tocd)')
###

index = robjects.r('index <- raster("E:/Education/MSc/thesis/UrbanExpansion/Data/UrbanExpansion/Predict(2004to2014)/Predict_2004To2014/model_epochs_BS256x256_05/Predict2014_model_epochs_BS256x256_05_WithoutTh.tif")')
boolean = robjects.r('boolean <- raster("E:/Education/MSc/thesis/UrbanExpansion/Data/UrbanExpansion/Predict(2004to2014)/ChangeDetection(Label)/Prediction_Label/Prediction_Label.tif")')
mask = robjects.r('mask <- raster("E:/Education/MSc/thesis/UrbanExpansion/Data/UrbanExpansion/Train(1994to2004)/Drivers/Data_Nan4000000/excl_both.tif")')

print(robjects.r('index'))
print(robjects.r('boolean'))
print(robjects.r('mask'))

print(robjects.r('unique(index)'))
print(robjects.r('unique(boolean)'))
print(robjects.r('unique(mask)'))

tocd = robjects.r('tocd <- TOC(index, boolean, mask, nthres = 100)')
robjects.r('plot(tocd)')
print(robjects.r('tocd'))


A = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Predict_2004To2014\ModelWithBatch_epochs_05\Predict2014_ModelWithBatch_epochs_05_WithoutTh.tif'))
unique, counts = np.unique(A, return_counts=True)
freq = np.asarray((unique, counts)).T

Min = robjects.r ('Min <- minValue(index) ')
thresholds = robjects.r ('thresholds <- c(0.1,0.2,0.3,0.5,0.6,0.7,0.8)')
#print((np.array(index)).shape)
# to3 = robjects.r('TOC(index, boolean, mask=mask, nthres=NULL, thres=NULL, NAval=0, P=NA, Q=NA, progress=FALSE)')
#
# # to = robjects.r('tocd <- TOC(index, boolean, mask, nthres = 100)')
# # pl = robjects.r('plot(tocd, main = "TOC curve")')
# #
# input()
#
# to2 = robjects.r('rocd <- ROC(index, boolean, mask, nthres = 100)')
# to2 = robjects.r('plot(rocd, main = "ROC curve")')

## thresholds can be defined by indicating the number of equal-interval thresholds
z = robjects.r('tocd <- roctable(index, boolean, mask, nthres=60000)')
x = robjects.r('tocd')
print(x)
## a vector of thresholds can also be used to define the thresholds
# robjects.r('thresholds <- seq(min(unique(index)), max(unique(index)) + 1, by = ceiling(max(unique(index))/10))')
# robjects.r('tocd <- TOC(index, boolean, mask, thres = thresholds)')
# robjects.r('tocd')
## all the unique values of the index object can be evaluated as thresholds (default option)
## Not run:
# tocd <- TOC(index, boolean, mask, progress = TRUE)
# tocd
# ## End(Not run)
# ## generate the TOC curve using non-spatial data (i.e., an object of class numeric)
# ## Not run:
# index <- getValues(index)
# boolean <- getValues(boolean)
# mask <- getValues(mask)
# tocd <- TOC(index, boolean, mask, nthres = 100)
## End(Not run)


#
#
# #ggplot2 = importr('ggplot2')
#importr('TOC')
# d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
# TOC = importr('TOC', robject_translations = d)
#
# bioc_url = urlopen('http://amsantac.co/software.html')
# string = ''.join(bioc_url.readlines())
#
# stringr_c = SignatureTranslatedAnonymousPackage(string, "stringr_c")



# utils.chooseCRANmirror(ind=1)

# packnames = ('A3')
# utils.install_packages(StrVector(packnames))
# A3 = importr('A3')

#
# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))
#
#TOC = importr('TOC')
# ggplot2 = importr('ggplot2')