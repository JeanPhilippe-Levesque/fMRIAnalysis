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

import scipy.ndimage as ndi
# Paramètres afin de changer la grandeur de l'image
rcParams['figure.figsize'] = 10, 12

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


def CroppedBrain(dataset, nbre_lignes, nbre_colonne, rowmax, rowmin, colmax, colmin):

    """ Cette fonction prend en entrée la liste de matrices (index [0]) créée par la fonction LstMatriceDicom
        sous forme de matrice d'intensité, ainsi que les valeurs limites des lignes et des colonnes du cerveau
        et retourme une liste de matrice dont la seule structure possèdant une intensité non-nulle est le cerveau.
        Les valeurs numériques limites sont extraites visuellements des images créées par le premier code dans
        lesquelles passent les données (IRMango).

        :params :
        :dataset : Liste des matrices (index [0]) créée par la fonction LstMatriceDicom (lst)
        :rowmax : Valeur maximale de la ligne d'où le cerveau est visible (int)
        :rowmin : Valeur minimale de la ligne d'où le cerveau est visible (int)
        :colmax : Valeur maximale de la colonne d'où le cerveau est visible (int)
        :colmin : Valeur minimale de la colonne d'où le cerveau est visible (int)

        :return :
        :dataset[0] : Liste de matrice correspondant au nombre d'image dicom dont seulement
                      le cerveau est visible (lst)
    """

    for matrices in range(len(dataset)):
        for row in range(nbre_colonne):
            for colum in range(nbre_lignes):
                if rowmin <= row <= rowmax and colmin <= colum <= colmax and dataset[matrices][row][colum] >= 800:
                    dataset[matrices][row][colum] = dataset[matrices][row][colum]
                else:
                    dataset[matrices][row][colum] = 0
    return dataset


def ImageFiltre(CroppedDataset, sigma):

    """ Cette fonction prend en entrée la liste de matrices créée par la fonction CroppedBrain , ainsi
        que la valeur de sigma voulu pour faire un filtre gaussien et retourne une image filtré selon
        la deviation standard choisi.

        :params :
        :CroppedDataset : Liste des matrices possédant seulement le cerveau (lst)
        :sigma : Valeur de la déviation standard (int)

        :return :
        :DataFiltre : Liste de matrice correspondant au nombre d'image dicom dont seulement
                      le cerveau est visible dont l'intensité des voxels est filtrée (lst)
    """

    DataFiltre = []
    for matrice in range(len(CroppedDataset)):
        DataFiltre.append(gaussian_filter(CroppedDataset[matrice], sigma))

    DataFiltre = np.array(DataFiltre)

    return DataFiltre


def RegroupementPixelsColonnes(lst, nbre_col, lignes):
    """ Cette fonction prend en entrée le resultat créé par la fonction LstMatriceDicom, ainsi
        que l'index d'une colonne dans la matrice. La fonction retourne une autre liste dont chaque index est
        une autre liste. Cette sous-liste regroupe la valeur d'intensité des pixels correspondant à l'index de
        la première liste pour chaque matric du dossier.

        Exemple afin de comprendre :

        lst = [[1, 2], [3, 4]] (M1), [[5, 6], [7, 8]] (M2), [[9, 10], [11, 12]] (M3) -> lst_test(lst, 0)

        lst_test(lst, 0) = [[1,5,9],[2,6,10]]

        :params :
        :lst : Variable étant une liste de matrice trouvée avec la fonction LstMatriceDicom (lst)

        :return :
        :chunk_lst : Liste 2D contenant l'intensité des pixels de chaque index de la colonne d'une matrices des images (lst)
    """

    lst_init=[]


    for columns in range(nbre_col):
        for matrix in range(len(lst)):
            lst_init.append(lst[matrix][lignes][columns])

    chunk_lst = np.array([lst_init[x:x+(len(lst))] for x in range(0, len(lst_init), (len(lst)))])
    # first_line_split = np.array_split(lst_init, len(lst_init))

    return chunk_lst


def RegroupementPixelsMAtrice(lst_ligne, nbre_col):

    """ Cette fonction prend en entrée une liste 2D contenant les valeurs de pixels de chaque index
        des colonnes de chaque matrice et retourne une matrice dont chaques élements de la matrice
        est une liste regroupant chaque valeur de pixels de l'index pour toutes les matrices.

        :params :
        :lst : Variable étant une liste de matrice trouvée avec la fonction LstMatriceDicom (lst)

        :return :
        :lst_matrice : Liste 3D contenant l'intensité des pixels pour chaque élément d'une matrices des images (lst)
    """
    lst_matrice = []

    for lines in range(nbre_col):
        lst_matrice.append(RegroupementPixelsColonnes(lst_ligne, nbre_col, lines))

    return lst_matrice


def StudentsTest(MatriceFiltreBaseline, MatriceFiltreCO2, nbre_lignes):

    resultats_Ttest = []

    for lignes in range(nbre_lignes):
        for colonnes in range(nbre_lignes):
            resultats_Ttest.append((stats.ttest_ind(MatriceFiltreBaseline[lignes][colonnes], MatriceFiltreCO2[lignes][colonnes])[-1]))


    Pmap = np.array([resultats_Ttest[x:x+100] for x in range(0, len(resultats_Ttest), 100)])

    return Pmap


def SortirIntensitePixel(MatricePmap, MatriceActivationCO2, MatriceBaseline):

    CarteIntensite = []

    for PmapColonnes in range(len(MatricePmap)):
        for Pmaplines in range(len(MatricePmap)):
            if MatricePmap[PmapColonnes][Pmaplines] > 0.0 and MatricePmap[PmapColonnes][Pmaplines] <=0.01:
                MatricePmap[PmapColonnes][Pmaplines] = MatricePmap[j][Pmaplines]
                CarteIntensite.append(MatriceActivationCO2[PmapColonnes][Pmaplines] - MatriceBaseline[PmapColonnes][Pmaplines])
            else:
                MatricePmap[PmapColonnes][Pmaplines] = 0
                CarteIntensite.append(0)

    CarteIntensite = np.array([CarteIntensite[x:x+100] for x in range(0, len(CarteIntensite), 100)])


    return CarteIntensite


def MakeTimeLineGraph(lst_matrice1, lst_matrice2, lst_matrice3, lst_matrice4, idxLigne, idxColl):

    #Élaboration des variables
    Pixel_value =[]
    nbre_image = []
    start = 0

    for matrix1 in range(len(lst_matrice1)):
        Pixel_value.append(lst_matrice1[matrix1][idxLigne][idxColl])

    for matrix2 in range(len(lst_matrice2)):
        Pixel_value.append(lst_matrice2[matrix2][idxLigne][idxColl])

    for matrix3 in range(len(lst_matrice3)):
        Pixel_value.append(lst_matrice3[matrix3][idxLigne][idxColl])

    for matrix4 in range(len(lst_matrice4)):
        Pixel_value.append(lst_matrice4[matrix4][idxLigne][idxColl])

    stop = len(Pixel_value)

    while start < stop:
        start += 1
        nbre_image.append(start)

    return Pixel_value, nbre_image


def ImageWithROI(lst_index, PixelsMap):
    y1, y2, x1, x2 = lst_index[0], lst_index[1], lst_index[-2], lst_index[-1]
    # lst_index[0], lst_index[1], lst_index[-2], lst_index[-1]

    plt.imshow(PixelsMap)
    plt.plot((x1,x2),(y2,y2), c='red')
    plt.plot((x1,x1),(y2,y1), c='red')
    plt.plot((x2,x1),(y1,y1), c='red')
    plt.plot((x2,x2),(y2,y1), c='red')
    plt.colorbar(shrink=0.70)
    plt.show()

    return


def MoyennePixelInROI(lst_matrice, DimMatrice, HighLine, LowLine,  LeftCol, RightCol):

    lst_pixel = []
    lst_inter = []

    for Images in range(len(lst_matrice)):
        for lines in range(HighLine, LowLine + 1):
            for colonnes in range(LeftCol, RightCol):
                lst_inter.append(lst_matrice[Images][lines][colonnes])

        Moyenne = np.mean(lst_inter)
        lst_pixel.append(Moyenne)
        lst_inter.clear()


    return lst_pixel


def ConcatenateMeanPixelLst(lst1, lst2, lst3, lst4):

    lst_concatenate = lst1 + lst2 + lst3 +lst4

    lst_NbreImage = []

    for idx in range(len(lst_concatenate)):
        lst_NbreImage.append(idx)


    return lst_concatenate, lst_NbreImage


def float_range(start, stop, step):
    """ Cette fonction permet de créer une liste de valeur ayant la
        en ayant la possibilité de choisir l'incrément nécessaire entre les
        valeurs. Par exemple, crée une liste de 0 à 1, avec des incréments de 0.1.
    """

    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)


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


def PmapPrecise(Pmap):

    for j in range(len(Pmap)):
        for g in range(len(Pmap)):
            if Pmap[j][g] > 0.0 and Pmap[j][g] <=0.005:
                Pmap[j][g] = Pmap[j][g]
            else:
                Pmap[j][g] = 0

    x = plt.imshow(Pmap)
    plt.colorbar(shrink=0.70)
    plt.show()
    return

# ------------------------- Importation des données
# Baseline
PathDicom_baseline1 = "8744 GRE SNAP DYNAMIC"

#Run 1 de CO2
PathDicom_CO2_1 = "8750 GRE SNAP DYNAMIC"

# Baseline 2
PathDicom_baseline2 = "8752 GRE SNAP DYNAMIC"

#Run 2 de CO2
PathDicom_CO2_2 = "8754 GRE SNAP DYNAMIC"

#ImageStructurel
PathDicom_SE = 'SE'

# ---------- Image de structurel -----------
lstSE, lignes, SpaceThick1 = LstMatriceDicom(PathDicom_SE)
plt.imshow(lstSE[1], cmap='gray')=


# --------------- Création des matrices ----------------------
lstBaseline_1, nbre_lignes_1, SpaceThick1 = LstMatriceDicom(PathDicom_baseline1)

lstCO2_1, nbre_lignes_2, SpaceThick2 = LstMatriceDicom(PathDicom_CO2_1)

lstBaseline_2, nbre_lignes_3, SpaceThick3 = LstMatriceDicom(PathDicom_baseline2)

lstCO2_2, nbre_lignes_4, SpaceThick4 = LstMatriceDicom(PathDicom_CO2_2)


# ---------------- Création des images moyennes ------------------
img_ref_Baseline_1 = ImageMoyenne(lstBaseline_1, nbre_lignes_1, nbre_lignes_1)
img_ref_Baseline_2 = ImageMoyenne(lstBaseline_2, nbre_lignes_2, nbre_lignes_2)


img_ref_CO2_1 = ImageMoyenne(lstCO2_1, nbre_lignes_3, nbre_lignes_3)
img_ref_CO2_2 = ImageMoyenne(lstCO2_2, nbre_lignes_4, nbre_lignes_4)


# ------------- fonctions -------------------

# # ---------------------------Baseline1 --------------------------
Baseline1_filter = ImageFiltre(lstBaseline_1,3)

Baseline1_cropped = CroppedBrain(Baseline1_filter, nbre_lignes_1, nbre_lignes_1, 40, 13, 71, 30)
baseline1_regroupement = RegroupementPixelsMAtrice(Baseline1_cropped, nbre_lignes_1)

# # --------------------------- CO2_1 --------------------------
CO2_1_filter = ImageFiltre(lstCO2_1,3)
CO2_1_cropped = CroppedBrain(CO2_1_filter, nbre_lignes_2, nbre_lignes_2, 40, 13, 71, 30)
CO2_1regroupement = RegroupementPixelsMAtrice(CO2_1_cropped, nbre_lignes_2)

# # ---------------------------Baseline_2 --------------------------
Baseline2_filter = ImageFiltre(lstBaseline_2,3)
Baseline3_cropped = CroppedBrain(Baseline2_filter, nbre_lignes_3, nbre_lignes_3, 40, 13, 71, 30)
baseline3_regroupement = RegroupementPixelsMAtrice(Baseline3_cropped, nbre_lignes_3)

# # --------------------------- CO2_2 --------------------------
CO2_2_filter = ImageFiltre(lstCO2_2,3)
CO2_2_cropped = CroppedBrain(CO2_2_filter, nbre_lignes_2, nbre_lignes_2, 40, 13, 71, 30)
CO2_r2egroupement = RegroupementPixelsMAtrice(CO2_1_cropped, nbre_lignes_2)


Student_Pmap = np.nan_to_num(StudentsTest(baseline1_regroupement, CO2_1regroupement, nbre_lignes_1), nan=0)


print('---------------- Carte P Values ----------------')
Student_graph = plt.imshow(Student_Pmap)
plt.colorbar(shrink=0.70)
plt.show()

PmapPrecise(Student_Pmap)

print('--------------- Carte dactivation --------------')
ActivationMap_1 = SortirIntensitePixel(Student_Pmap, img_ref_CO2_1, img_ref_Baseline_1)
plt.imshow(ActivationMap_1)
plt.colorbar(shrink=0.70)
plt.show()


lst_index = list(map(int, input("Entrée valeur ligne haute, ligne basse, colonne gauche, colonne droite (sans utiliser ,): ").split()))

ROI_baseline_1 = MoyennePixelInROI(lstBaseline_1, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
ROI_CO2_1 = MoyennePixelInROI(lstCO2_1, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
ROI_baseline_2 = MoyennePixelInROI(lstBaseline_2, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
ROI_CO2_2 = MoyennePixelInROI(lstCO2_2, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])

ImageWithROI(lst_index, ActivationMap_1)

# -------------- créer une fonction pour ca --------------
Intesite_moyenne, Nmbre_image = ConcatenateMeanPixelLst(ROI_baseline_1, ROI_CO2_1, ROI_baseline_2, ROI_CO2_2)
yhat = ndi.gaussian_filter(Intesite_moyenne, 4)

Fin_premiere_serie = len(lstBaseline_1)
Debut_deuxieme_serie = Fin_premiere_serie + 1
Fin_deuxieme_serie = Fin_premiere_serie + len(lstCO2_1)
Debut_troisieme_serie = Fin_deuxieme_serie + 1
Fin_troisieme_serie = Fin_deuxieme_serie + len(lstBaseline_2)
Debut_quatrieme_serie = Fin_troisieme_serie + 1
Fin_quatrieme_serie = Fin_troisieme_serie + len(lstCO2_2)

Val_min = np.min(Intesite_moyenne) - 30



ax13 = plt.subplot()
ax13.plot(Nmbre_image, Intesite_moyenne,label='Données brutes')
ax13.plot(Nmbre_image, yhat, color='r', label='Données smooths')
plt.legend()
ax13.axvspan(Debut_deuxieme_serie, Fin_deuxieme_serie, color='silver')
ax13.axvspan(Debut_quatrieme_serie, Fin_quatrieme_serie, color='silver')
ax13.text(10, Val_min, 'Baseline 1',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
ax13.text(52, Val_min, 'CO2',
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10})
ax13.text(86, Val_min, 'Baseline 2',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
ax13.text(127, Val_min, 'CO2',
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10})
plt.show()



# ----------------------- Mesure des déplacements de la tête --------------------------
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

# fig = plt.figure()
#
# ax = plt.subplot(111)
# ax.plot(NbreImageBaseline_1, DeplacementBaseline_1_X, color='b', label='Déplacements en y')
# ax.plot(NbreImageBaseline_1, DeplacementBaseline_1_Y, color='orange', label='Déplacements en x')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.savefig('CM_baseline_1.png')
# plt.legend()
# ax.tick_params(direction='in')
# plt.show()
#
# ax2 = plt.subplot(111)
# ax2.plot(NbreCO2_1, DeplacementCO2_1_X, color='b', label='Déplacements en x')
# ax2.plot(NbreCO2_1, DeplacementCO2_1_Y, color='orange', label='Déplacements en y')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.legend()
# ax2.tick_params(direction='in')
# plt.show()
#
# ax3 = plt.subplot(111)
# ax3.plot(NbreImageBaseline_2, DeplacementBaseline_2_X, color='b', label='Déplacements en x')
# ax3.plot(NbreImageBaseline_2, DeplacementBaseline_2_Y, color='orange', label='Déplacements en y')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.legend()
# ax3.tick_params(direction='in')
# plt.show()
#
#
# ax4 = plt.subplot(111)
# ax4.plot(NbreCO2_2, DeplacementCO2_2_X, color='b', label='Déplacements en x')
# ax4.plot(NbreCO2_2, DeplacementCO2_2_Y, color='orange', label='Déplacements en y')
# plt.xlabel('Image [-]')
# plt.ylabel('Déplacements [mm]')
# plt.legend()
# ax4.tick_params(direction='in')
# plt.show()

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
