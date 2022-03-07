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
import sys
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
                MatricePmap[PmapColonnes][Pmaplines] = MatricePmap[PmapColonnes][Pmaplines]
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

# ------------------------- Importation des données
# Baseline
PathDicom_baseline1 = "8744 GRE SNAP DYNAMIC"

#Run 1 de CO2
PathDicom_CO2_1 = "8750 GRE SNAP DYNAMIC"

# Baseline 2
PathDicom_baseline2 = "8752 GRE SNAP DYNAMIC"

#Run 2 de CO2
PathDicom_CO2_2 = "8754 GRE SNAP DYNAMIC"

SE = "SE"

rowmax = 40

rowmin = 13

colmax = 71

colmin = 30

# --------------- Création des matrices ----------------------
lstBaseline_1, nbre_lignes_1, SpaceThick1 = LstMatriceDicom(PathDicom_baseline1)

lstCO2_1, nbre_lignes_2, SpaceThick2 = LstMatriceDicom(PathDicom_CO2_1)

lstBaseline_2, nbre_lignes_3, SpaceThick3 = LstMatriceDicom(PathDicom_baseline2)

lstCO2_2, nbre_lignes_4, SpaceThick4 = LstMatriceDicom(PathDicom_CO2_2)

SE, nbre_lignes_5, SpaceThick5 = LstMatriceDicom(SE)
# ----------- Assignation des images moyennes -----------------
img_ref_Baseline_1 = ImageMoyenne(lstBaseline_1, nbre_lignes_1, nbre_lignes_1)

img_ref_CO2 = ImageMoyenne(lstCO2_1, nbre_lignes_2, nbre_lignes_2)


# ------------- TEST des fonctions -------------------
# # ---------------------------Baseline1 --------------------------
# x1 = CroppedBrain(lstBaseline_1, nbre_lignes_1, nbre_lignes_1, rowmax, rowmin, colmax, colmin)
# y1 = ImageFiltre(x1,3)
# baseline_regroupement = RegroupementPixelsMAtrice(x1, nbre_lignes_1)
#
#
# # # --------------------------- CO2 1 --------------------------
# x2 = CroppedBrain(lstCO2_1, nbre_lignes_2, nbre_lignes_2, rowmax, rowmin, colmax, colmin)
# y2 = ImageFiltre(lstCO2_1,3)
# CO2_regroupement = RegroupementPixelsMAtrice(x2, nbre_lignes_2)

Baseline1_filter = ImageFiltre(lstBaseline_1,3)
Baseline1_cropped = CroppedBrain(Baseline1_filter, nbre_lignes_1, nbre_lignes_1, 40, 13, 71, 30)
baseline1_regroupement = RegroupementPixelsMAtrice(Baseline1_cropped, nbre_lignes_1)

# # --------------------------- CO2_1 --------------------------
CO2_1_filter = ImageFiltre(lstCO2_1,3)
CO2_1_cropped = CroppedBrain(CO2_1_filter, nbre_lignes_2, nbre_lignes_2, 40, 13, 71, 30)
CO2_1regroupement = RegroupementPixelsMAtrice(CO2_1_cropped, nbre_lignes_2)

Student_Pmap = np.nan_to_num(StudentsTest(baseline1_regroupement, CO2_1regroupement, nbre_lignes_1), nan=0)

print('---------------- Carte P Values ----------------')
Student_graph = plt.imshow(Student_Pmap)
plt.colorbar(shrink=0.70)
plt.show()

print('--------------- Carte dactivation --------------')

rslt = SortirIntensitePixel(Student_Pmap, img_ref_CO2, img_ref_Baseline_1)
rslt[rslt==0] = np.nan
plt.imshow(SE[1], cmap='gray')
plt.imshow(rslt, cmap='seismic')
# plt.colorbar(shrink=0.70)
plt.axis('off')
plt.show()
#
# lst_index = list(map(int, input("Entrée valeur ligne haute, ligne basse, colonne gauche, colonne droite (sans utiliser ,): ").split()))
# # for idx in range(len(lst_index)):
# #     if type(lst_index[idx]) == int:
# #         continue
# #     else:
# #         sys.exit("Recommence")
#
# ROI_baseline_1 = MoyennePixelInROI(lstBaseline_1, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
# ROI_CO2_1 = MoyennePixelInROI(lstCO2_1, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
# ROI_baseline_2 = MoyennePixelInROI(lstBaseline_2, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
# ROI_CO2_2 = MoyennePixelInROI(lstCO2_2, 100, lst_index[0], lst_index[1], lst_index[2], lst_index[-1])
#
#
# Intesite_moyenne, Nmbre_image = ConcatenateMeanPixelLst(ROI_baseline_1, ROI_CO2_1, ROI_baseline_2, ROI_CO2_2)
# yhat = ndi.gaussian_filter(Intesite_moyenne, 4)
#
# ImageWithROI(lst_index, rslt)
#
# # -------------- créer une fonction pour ca --------------
#
# Fin_premiere_serie = len(lstBaseline_1)
# Debut_deuxieme_serie = Fin_premiere_serie + 1
# Fin_deuxieme_serie = limBL1 + len(lstCO2_1)
# Debut_troisieme_serie = Fin_deuxieme_serie + 1
# Fin_troisieme_serie = Fin_deuxieme_serie + len(lstBaseline_2)
# Debut_quatrieme_serie = Fin_troisieme_serie + 1
# Fin_quatrieme_serie = Fin_troisieme_serie + len(lstCO2_2)
#
#
#
# ax13 = plt.subplot()
# ax13.plot(Nmbre_image, Intesite_moyenne)
# ax13.plot(Nmbre_image, yhat, color='r')
# ax13.axvspan(Debut_deuxieme_serie, Fin_deuxieme_serie, color='silver')
# ax13.axvspan(Debut_quatrieme_serie, Fin_quatrieme_serie, color='silver')
# ax13.text(10, 550, 'Baseline 1',
#         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
# ax13.text(52, 550, 'CO2',
#         bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10})
# ax13.text(86, 550, 'Baseline 2',
#         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
# ax13.text(127, 550, 'CO2',
#         bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10})
# plt.show()
