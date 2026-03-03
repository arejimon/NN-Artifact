import itk
import SimpleITK as sitk
import numpy as np
import sys

# --------------------------------------------------

def read_image(img_dir):
    # Set up names generator
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    dicomIO = itk.GDCMImageIO.New()

    # Read fixed into 3D volume
    namesGenerator.SetDirectory(img_dir)
    seriesUID = namesGenerator.GetSeriesUIDs()

    if len(seriesUID) < 1:
        print("No DICOMs in: " + img_dir)
        sys.exit(1)

    fileNames = namesGenerator.GetFileNames(seriesUID[0])

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(fileNames)
    fixed = reader.Execute()
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)

    fixed = sitk.DICOMOrient(fixed, 'RPI')
    
    print(fixed.GetSize())

    return fixed

def load_sitk_img(sitkVol):
    img = sitk.GetArrayFromImage(sitkVol)
    
    if np.max(img) > 0.0:
        img = (img - np.mean(img)) / np.std(img)
    
    return np.transpose(img, (1,2,0))

def sitk_to_npy(sitkVol):
    img = sitk.GetArrayFromImage(sitkVol)
    return np.transpose(img, (1,2,0))

def itk_to_sitk(itkImg):
    npyImg = itk.array_from_image(itkImg)
    sitkImg = sitk.GetImageFromArray(npyImg)
    
    sitkImg.SetOrigin(tuple(itkImg.GetOrigin()))
    sitkImg.SetDirection(tuple(itk.GetArrayFromMatrix(itkImg.GetDirection()).flatten()))
    sitkImg.SetSpacing(tuple(itkImg.GetSpacing()))
    
    return sitkImg

def sitk_to_itk(sitkImg):
    npyImg = sitk.GetArrayFromImage(sitkImg)
    itkImg = itk.image_from_array(npyImg)
    
    itkImg.SetOrigin(sitkImg.GetOrigin())
    itkImg.SetDirection(itk.matrix_from_array(np.reshape(sitkImg.GetDirection(),(3,3))))
    itkImg.SetSpacing(sitkImg.GetSpacing())
    
    return itkImg

def npy_to_sitk(img, ref_img):
    itkImg = sitk.GetImageFromArray(img)
    itkImg.SetSpacing(ref_img.GetSpacing())
    itkImg.SetOrigin(ref_img.GetOrigin())
    itkImg.SetDirection(ref_img.GetDirection())
    
    return itkImg