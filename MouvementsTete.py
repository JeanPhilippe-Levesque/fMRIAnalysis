import imageio as io
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import numpy as np
from scipy import ndimage
import pydicom
import os
import numpy as np
from pylab import rcParams
from matplotlib import pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy import misc
import cv2 as cv
import csv
import matplotlib.image as mpimg
import decimal
import statistics as st
from skimage.feature import register_translation
from image_registration import chi2_shift
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift
from PIL import Image



# ------------------------ BEGIN OF FUNCTIONS ----------------------------

# ---------------- Fonction 1 -----------------
def LstMatriceDicom(PathDicom):

    """ Cette fonction prend en entrée le chemin du dossier contenant les fichiers dicom
        et retourne une liste de matrices d'intensité correspondant à l'image acquise.
        Cette fonction est utilisée afin d'extraire les images dicoms en données analysable
        à l'aide de fonction.

        :params :
        :PathDicom : Variable contenant le project path du dossier voulu

        :return :
        :ArrayDicom : Liste de matrices des images  (lst)
        :ConstPixelDims[1] : Nombre de lignes dans les matrices (int)
        :ConstPixelDims[-1] :Nombre de colonnes dans les matrices (int)
    """

    # create an empty list qui va acceuillir les fichiers DICOM
    lstFilesDCM = []


    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

                # On va chercher les informations concernant les images DICOM que l'on vient d'importer
    RefDs = pydicom.read_file(lstFilesDCM[0])


    # On va chercher les dimensions des colonnes et des lignes pour avoir (nombre de fichier, lignesm colonnes)
    ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # Met les longueurs des images en mm
    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])


    # On créer un array de matrice vide possèdant les dimensions de 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
        #store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

    return ArrayDicom, ConstPixelDims[1], ConstPixelDims[-1]

# ---------------- Fonction 2 -----------------
def ImageMoyenne(lstmatrice, nbre_lignes_1, nbre_lignes_2):

    """ Cette fonction prend en entrée la liste de matrices (index [0]) créée par la fonction LstMatriceDicom
        et retourne une image dont chaque pixel est la moyenne de toute les images de la liste des matrices.

        :params :
        :lstmatrice : Liste des matrices (lst)

        :return :
        :ImageReconstruite : Matrice dont les valeurs correspondent à l'intensité moyenne de chaque pixel (lst)

    """
    # On crée une variable (matrice) vide, les dimensions doivent être ajuster en fonction du nombre
    # de pixel de la serie DICOM
    ImageReconstruite = np.zeros((nbre_lignes_1, nbre_lignes_2))

    # On additionne toutes les matrices ensembles
    for i in range(len(lstmatrice)):
        ImageReconstruite += lstmatrice[i]

    # On fait la moyenne avec le nombre de fichiers totale
    ImageReconstruite = ImageReconstruite / len(lstmatrice)

    return ImageReconstruite


# ---------------- Fonction 3 -----------------
def float_range(start, stop, step):
    """ Cette fonction permet de créer une liste de valeur ayant la
        en ayant la possibilité de choisir l'incrément nécessaire entre les
        valeurs. Par exemple, crée une liste de 0 à 1, avec des incréments de 0.1.
    """

    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)


# ---------------- Fonction 4 -----------------
def MesureDeplacement(matrice_moyenne,lst_matrice, GrandeurPixel):


    cm_init = ndimage.center_of_mass(matrice_moyenne)

    position_x = []
    position_y = []
    deplacement_x_in_mm = []
    deplacement_y_in_mm = []

    for i in range(len(lst_matrice)):
        matrice_seconde = lst_matrice[i]
        cm_seconde = ndimage.center_of_mass(matrice_seconde)
        position_x.append(cm_seconde[0])
        position_y.append(cm_seconde[-1])
        deplacement_x_in_mm.append((position_x[-1] - cm_init[0]) * GrandeurPixel)
        deplacement_y_in_mm.append((position_y[-1] - cm_init[-1]) * GrandeurPixel)

    return deplacement_x_in_mm, deplacement_y_in_mm


# ---------------- Fonction 5 -----------------
def chi2_shift_from_folder(img_ref, lstImageDCM, PxlSpacing):
    """ La méthode chi2_shift trouve le décalage entre deux images en utilisant la DFT
        upsampling method étant une 2D rigid image registration. La fonction prend en entrée
        une image de référence étant l'image moyenne produite pendant les différentes runs
        et le path contenant les images png sans infos des  """
    img_moving = []
    Moving_x = []
    Moving_y = []
    noise = 0.1

    img_refarray = np.array(img_ref)

    for files in range(len(lstImageDCM)):
        img_moving = lstImageDCM[files]
        x_off, y_off, ex_off, ey_off = chi2_shift(img_ref, img_moving, noise, return_error=True, upsample_factor='auto')
        Moving_x.append(x_off * PxlSpacing)
        Moving_y.append(y_off * PxlSpacing)

    return Moving_x, Moving_y


# ---------------- Fonction 6 -----------------
def CrossCorrelation_from_folder(img_ref,lstImageDCM, PxlSpacing):

    """ La méthode chi2_shift trouve le décalage entre deux images en utilisant la DFT
        upsampling method étant une 2D rigid image registration. La fonction prend en entrée
        une image de référence étant l'image moyenne produite pendant les différentes runs
        et le path contenant les images png sans infos des  """

    Moving_x = []
    Moving_y = []

    for files in range(len(lstImageDCM)):
                img_moving = lstImageDCM[files]
                img_movingarray = np.array(img_moving)
                x_off, y_off = cross_correlation_shifts(img_ref, img_moving)
                Moving_x.append(x_off * PxlSpacing)
                Moving_y.append(y_off * PxlSpacing)


    return Moving_x, Moving_y

# ------------------------ END OF FUNCTIONS ----------------------------




# --------------- Importation des données --------------------
PathDicom_baseline1 = "8744 GRE SNAP DYNAMIC"

PathDicom_CO2_1 = "8750 GRE SNAP DYNAMIC"

PathDicom_baseline2 = "8752 GRE SNAP DYNAMIC"

PathDicom_CO2_2 = "8754 GRE SNAP DYNAMIC"

# --------------- Création des matrices ----------------------
lstBaseline_1, nbre_lignes_1, SpaceThick1 = LstMatriceDicom(PathDicom_baseline1)

lstCO2_1, nbre_lignes_2, SpaceThick2 = LstMatriceDicom(PathDicom_CO2_1)

lstBaseline_2, nbre_lignes_3, SpaceThick3 = LstMatriceDicom(PathDicom_baseline2)

lstCO2_2, nbre_lignes_4, SpaceThick4 = LstMatriceDicom(PathDicom_CO2_2)

print(len(lstBaseline_1))

# --------------- Création des images  ----------------------
img_ref_Baseline_1 = ImageMoyenne(lstBaseline_1, nbre_lignes_1, nbre_lignes_1)
img_ref_Baseline_2 = ImageMoyenne(lstBaseline_2, nbre_lignes_2, nbre_lignes_2)

img_ref_CO2_1 = ImageMoyenne(lstCO2_1, nbre_lignes_3, nbre_lignes_3)
img_ref_CO2_2 = ImageMoyenne(lstCO2_2, nbre_lignes_4, nbre_lignes_4)


folder = '/Users/jplevesque/Desktop/Code_fMRI/GRE_SNAP_CO2_1'

# ------------------------------------------------------------
print('Début des résultats avec technique Centre de Masse')
print('Création des listes de matrices')
print(' ')

DeplacementBaseline_1_X, DeplacementBaseline_1_Y = MesureDeplacement(img_ref_Baseline_1,lstBaseline_1, 0.250)
DeplacementCO2_1_X, DeplacementCO2_1_Y = MesureDeplacement(img_ref_CO2_1,lstCO2_1, 0.250)
DeplacementBaseline_2_X, DeplacementBaseline_2_Y = MesureDeplacement(img_ref_Baseline_2,lstBaseline_2, 0.250)
DeplacementCO2_2_X, DeplacementCO2_2_Y = MesureDeplacement(img_ref_CO2_2,lstCO2_2, 0.250)


print('Déplacements mesurés')
print(' ')

NbreImageBaseline_1 = list(float_range(0, len(Baseline_1[0]), '1'))
NbreCO2_1 = list(float_range(0, len(CO2_1[0]), '1'))
NbreImageBaseline_2 = list(float_range(0, len(Baseline_2[0]), '1'))
NbreCO2_2 = list(float_range(0, len(CO2_2[0]), '1'))


print('Graphiques')
print(' ')

rcParams['figure.figsize'] = 10, 12

fig = plt.figure()

ax = plt.subplot(111)
ax.plot(NbreImageBaseline_1, DeplacementBaseline_1_X, color='b', label='Déplacements en y')
ax.plot(NbreImageBaseline_1, DeplacementBaseline_1_Y, color='orange', label='Déplacements en x')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.savefig('CM_baseline_1.png')
plt.legend()
ax.tick_params(direction='in')
plt.show()

ax2 = plt.subplot(111)
ax2.plot(NbreCO2_1, DeplacementCO2_1_X, color='b', label='Déplacements en x')
ax2.plot(NbreCO2_1, DeplacementCO2_1_Y, color='orange', label='Déplacements en y')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.legend()
ax2.tick_params(direction='in')
plt.show()

ax3 = plt.subplot(111)
ax3.plot(NbreImageBaseline_2, DeplacementBaseline_2_X, color='b', label='Déplacements en x')
ax3.plot(NbreImageBaseline_2, DeplacementBaseline_2_Y, color='orange', label='Déplacements en y')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.legend()
ax3.tick_params(direction='in')
plt.show()


ax4 = plt.subplot(111)
ax4.plot(NbreCO2_2, DeplacementCO2_2_X, color='b', label='Déplacements en x')
ax4.plot(NbreCO2_2, DeplacementCO2_2_Y, color='orange', label='Déplacements en y')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.legend()
ax4.tick_params(direction='in')
plt.show()

print('Tableau récapitulatif')
print(' ')
print('# Image\t\tMoyenne en y(mm)\tMoyenne en x(mm)\t')
print('----------------------------------------------------------')
print( "|Baseline 1"f"\t{round(np.mean(DeplacementBaseline_1_X),5)}\t\t{round(np.mean(DeplacementBaseline_1_Y),5)}\t\t|")
print( "|CO2 1"f"\t\t{round(np.mean(DeplacementCO2_1_X),5)}\t\t\t{round(np.mean(DeplacementCO2_1_Y),5)}\t|")
print( "|Baseline 2"f"\t{round(np.mean(DeplacementBaseline_2_X),5)}\t\t\t{round(np.mean(DeplacementBaseline_2_Y),5)}\t|")
print( "|CO2 2"f"\t\t{round(np.mean(DeplacementCO2_2_X),5)}\t\t\t{round(np.mean(DeplacementCO2_2_Y),5)}\t|")
print('----------------------------------------------------------')

print('')
print('Fin des résultats avec technique Centre de Masse')
print('-------------------------------------------------')
print('')
print("Début des résultats avec technique décalage d'image")


""" Cette technique utilise la fonctionn skimage.registration.phase_cross_correlation (reference_image,
                                                                                       moving_image, *,
                                                                                       upsample_factor=1,
                                                                                       space='real',
                                                                                       return_error=True,
                                                                                       reference_mask=None,
                                                                                       moving_mask=None,
                                                                                       overlap_ratio=0.3,
                                                                                       normalization='phase').
    La fonction
    prend en entrée une image de référence et une autre image et permet de trouver le décalage
    entre les deux images. Le décalage est donné par un vecteur de translation en pixel selon l'ordre
    (Z,Y,X). En sachant la grandeur d'un pixel, on peut trouver la valeur métrique du décalage et
    la comparée avec les valeurs obtenues avec la technique du Centre de Masse. Il est conseiller,
    pour des images à fort bruit de ne pas utiliser l'option normalisation de la fonction.

"""

print('')
print('')
print('')

# -------------- Calcul des décalages pour les 4 runs différentes ---------------

Moving_x_1, Moving_y_1 = chi2_shift_from_folder(img_ref_Baseline_1, lstBaseline_1, 0.25)
CC_moving_X_1, CC_moving_Y_1 = CrossCorrelation_from_folder(img_ref_Baseline_1, lstBaseline_1, 0.25)

Moving_x_2, Moving_y_2 = chi2_shift_from_folder(img_ref_Baseline_2, lstBaseline_2, 0.25)
CC_moving_X_2, CC_moving_Y_2 = CrossCorrelation_from_folder(img_ref_Baseline_2, lstBaseline_2, 0.25)

Moving_x_3, Moving_y_3 = chi2_shift_from_folder(img_ref_CO2_1, lstCO2_1, 0.25)
CC_moving_X_3, CC_moving_Y_3 = CrossCorrelation_from_folder(img_ref_CO2_1, lstCO2_1, 0.25)

Moving_x_4, Moving_y_4 = chi2_shift_from_folder(img_ref_CO2_2, lstCO2_2, 0.25)
CC_moving_X_4, CC_moving_Y_4 = CrossCorrelation_from_folder(img_ref_CO2_2, lstCO2_2, 0.25)

print('-------------------- chi2_shift method -------------------')
# print('-----------------------------------------' )
# print('Déplacement en X    |    Déplacement en Y' )
# print('-----------------------------------------' )
# for i in range(len(Moving_x)):
#     print('|',round(Moving_x[i],4),'         |            ', round(Moving_y[i],4),'|')
# print('-----------------------------------------' )
# print('Moyenne en X = ',round(np.mean(Moving_x),4),', Moyenne en Y = ',round(np.mean(Moving_y),4))
#
# x1 = list(range(0, len(CC_moving_X)))

print('Tableau récapitulatif')
print(' ')
print('# Image\t\tMoyenne en y(mm)\tMoyenne en x(mm)\t')
print('----------------------------------------------------------')
print( "|Baseline 1"f"\t{round(np.mean(Moving_x_1),5)}\t\t\t{round(np.mean(Moving_y_1),5)}\t\t|")
print( "|CO2 1"f"\t\t{round(np.mean(Moving_x_3),5)}\t\t\t{round(np.mean(Moving_y_3),5)}\t\t|")
print( "|Baseline 2"f"\t{round(np.mean(Moving_x_2),5)}\t\t\t{round(np.mean(Moving_y_2),5)}\t|")
print( "|CO2 2"f"\t\t{round(np.mean(Moving_x_4),5)}\t\t\t{round(np.mean(Moving_y_4),5)}\t\t|")
print('----------------------------------------------------------')

print('')
print('')
print('')

print('-------------------- cross correlations shifts method -------------------')

# print('-----------------------------------------' )
# print('Déplacement en X    |    Déplacement en Y' )
# print('-----------------------------------------' )
# for i in range(len(CC_moving_X)):
#     print('|',round(CC_moving_X[i],4),'         |            ', round(CC_moving_Y[i],4),'|')
# print('-----------------------------------------' )
# print('Moyenne en X = ',round(np.mean(CC_moving_X),4),', Moyenne en Y = ',round(np.mean(CC_moving_Y),4))

print('Tableau récapitulatif')
print(' ')
print('# Image\t\tMoyenne en y(mm)\tMoyenne en x(mm)\t')
print('---------------------------------------------------------')
print( "|Baseline 1"f"\t{round(np.mean(CC_moving_X_1),5)}\t\t\t{round(np.mean(CC_moving_Y_1),5)}\t\t|")
print( "|CO2 1"f"\t\t{round(np.mean(CC_moving_X_3),5)}\t\t\t{round(np.mean(CC_moving_Y_3),5)}\t|")
print( "|Baseline 2"f"\t{round(np.mean(CC_moving_X_2),5)}\t\t\t{round(np.mean(CC_moving_Y_2),5)}\t\t|")
print( "|CO2 2"f"\t\t{round(np.mean(CC_moving_X_4),5)}\t\t\t{round(np.mean(CC_moving_Y_4),5)}\t|")
print('---------------------------------------------------------')

Baseline1_nb = [i for i in range(39)]
Baseline2_nb = [i for i in range(38)]
CO21_nb = [i for i in range(34)]
CO22_nb = [i for i in range(33)]

print(len(Baseline2_nb))
print(len(CC_moving_X_3))

rcParams['figure.figsize'] = 13, 16
# ---------------- Ligne 1 -----------------
ax5 = plt.subplot(421)
ax5.plot(Baseline1_nb, Moving_x_1, color='b', label='Déplacements en y C2_S')
ax5.plot(Baseline1_nb, CC_moving_X_1, color='r', label='Déplacements en y C_C')
ax5.plot(Baseline1_nb, DeplacementBaseline_1_X, color='g', label='Déplacements en y CM')
# plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.title('Déplacement en X pour Baseline_1')
plt.legend()
ax5.tick_params(direction='in')

ax6 = plt.subplot(422)
ax6.plot(Baseline1_nb, Moving_y_1, color='b', label='Déplacements en x chi2_shift')
ax6.plot(Baseline1_nb, CC_moving_Y_1, color='r', label='Déplacements en x cc')
ax6.plot(Baseline1_nb, DeplacementBaseline_1_Y, color='g', label='Déplacements en x cM')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
plt.title('Déplacement en Y pour Baseline_1')
plt.legend()
ax6.tick_params(direction='in')

#
# # ---------------- Ligne 2 -----------------
ax7 = plt.subplot(423)
ax7.plot(Baseline2_nb, Moving_x_2, color='b', label='Déplacements en y C2_S')
ax7.plot(Baseline2_nb, CC_moving_X_2, color='r', label='Déplacements en y C_C')
ax7.plot(Baseline2_nb, DeplacementBaseline_2_X, color='g', label='Déplacements en y CM')
# plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.title('Déplacement en X pour Baseline_2')
plt.legend()
ax7.tick_params(direction='in')

ax8 = plt.subplot(424)
ax8.plot(Baseline2_nb, Moving_y_2, color='b', label='Déplacements en x chi2_shift')
ax8.plot(Baseline2_nb, CC_moving_Y_2, color='r', label='Déplacements en x cc')
ax8.plot(Baseline2_nb, DeplacementBaseline_2_Y, color='g', label='Déplacements en x cM')
plt.title('Déplacement en Y pour Baseline_2')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.savefig('Déplacement en Y.png')
plt.legend()
ax8.tick_params(direction='in')

#
# # ---------------- Ligne 3 -----------------
ax9 = plt.subplot(425)
ax9.plot(CO21_nb, Moving_x_3, color='b', label='Déplacements en x chi2_shift')
ax9.plot(CO21_nb, CC_moving_X_3, color='r', label='Déplacements en x cc')
ax9.plot(CO21_nb, DeplacementCO2_1_X, color='g', label='Déplacements en x cc')
plt.title('Déplacement en X pour CO2 (1)')
# plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
# plt.savefig('Déplacement en Y.png')
plt.legend()
ax9.tick_params(direction='in')

ax10 = plt.subplot(426)
ax10.plot(CO21_nb, Moving_y_3, color='b', label='Déplacements en x chi2_shift')
ax10.plot(CO21_nb, CC_moving_Y_3, color='r', label='Déplacements en x cc')
ax10.plot(CO21_nb, DeplacementCO2_1_Y, color='g', label='Déplacements en x cc')
plt.title('Déplacement en Y pour CO2 (1)')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.savefig('Déplacement en Y.png')
plt.legend()
ax10.tick_params(direction='in')
#
# # ---------------- Ligne 4 -----------------
#
ax11 = plt.subplot(427)
ax11.plot(CO22_nb, Moving_x_4, color='b', label='Déplacements en y C2_S')
ax11.plot(CO22_nb, CC_moving_X_4, color='r', label='Déplacements en y C_C')
ax11.plot(CO22_nb, DeplacementCO2_2_X, color='g', label='Déplacements en y C_C')
plt.title('Déplacement en Y pour CO2 (2)')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
plt.legend()
ax11.tick_params(direction='in')

ax12 = plt.subplot(428)
ax12.plot(CO22_nb, Moving_y_4, color='b', label='Déplacements en x chi2_shift')
ax12.plot(CO22_nb, CC_moving_Y_4, color='r', label='Déplacements en x cc')
ax12.plot(CO22_nb, DeplacementCO2_2_Y, color='g', label='Déplacements en x cc')
plt.title('Déplacement en X pour CO2 (2)')
plt.xlabel('Image [-]')
plt.ylabel('Déplacements [mm]')
# plt.savefig('Déplacement en Y.png')
plt.legend()
ax12.tick_params(direction='in')
