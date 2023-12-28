import os
from PIL import Image
import pydicom
import numpy as np
import nibabel as nib
from mayavi import mlab
import cv2

def dicom_to_png(dicom_path, output_folder):
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_data = dicom_data.pixel_array
    image_data = (pixel_data / np.max(pixel_data) * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    os.makedirs(output_folder, exist_ok=True)
    png_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(dicom_path))[0]}.png")
    image.save(png_path, "PNG")
    print(f"Conversion complete. PNG saved at: {png_path}")

def convert_folder_to_png(input_folder, output_folder):
    dicom_files = [f for f in os.listdir(input_folder) if f.endswith(".dcm")]
    for dicom_file in dicom_files:
        dicom_path = os.path.join(input_folder, dicom_file)
        dicom_to_png(dicom_path, output_folder)

def remove_black_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, file))
                if np.max(ds.pixel_array) == 0:
                    os.remove(os.path.join(root, file))
                    print(f"Removed {os.path.join(root, file)}")

def croppng(folderinput, folderoutput):
    minx=600
    miny=600
    maxx=0
    maxy=0
    files = [file for file in os.listdir(folderinput) if file.endswith('.png') and file.startswith('Image-')]

    if not files:
        print(f"No PNG files found in {folderinput}")
        return

    # Iterate over the files to determine the range
    for i in files:
        #name=folderinput +'/Image-'+str(i)+'.png'
        name = os.path.join(folderinput, i)
        # load image as grayscale
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape
        thresh[hh-3:hh, 0:ww] = 0
        white = np.where(thresh==255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        if xmin<minx:
            minx=xmin
        if xmax>maxx:
            maxx=xmax
        if ymin<miny:
            miny=ymin
        if ymax>maxy:
            maxy=ymax
        print(xmin,xmax,ymin,ymax)
        #savethresh=folderouput + '/Image-'+str(i)+'.png'
        #cv2.imwrite(savethresh, thresh)
    for j in files:
        #name=folderinput+'/Image-'+str(j)+'.png'
        name = os.path.join(folderinput, j)
        # load image as grayscale
        img = cv2.imread(name)
        crop = img[miny:maxy+3, minx:maxx]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(crop)
        img_resized = pil_image.resize((64, 64), Image.LANCZOS)
        
        #img_resized = crop.resize((64, 64), Image.LANCZOS)
        savecrop = os.path.join(folderoutput, j)
        img_resized_np = np.array(img_resized)
        cv2.imwrite(savecrop, img_resized_np)

# def resize_images(input_folder, output_folder, size=(64, 64)):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)

#         img = Image.open(input_path)
#         img_resized = img.resize(size, Image.LANCZOS)
#         img_resized.save(output_path)

def png_series_to_nifti(png_folder, nifti_path):
    png_files = sorted([
                f for f in os.listdir(png_folder) if f.endswith(".png")
            ], key=lambda x: int(''.join(filter(str.isdigit, x))))

    first_image_path = os.path.join(png_folder, png_files[0])
    first_image = Image.open(first_image_path)
    img_shape = (len(png_files), first_image.height, first_image.width)

    volume_data = np.zeros(img_shape, dtype=np.uint8)
    for i, png_file in enumerate(png_files):
        image_path = os.path.join(png_folder, png_file)
        img = Image.open(image_path)
        print(img)
        volume_data[i, :, :] = np.array(img)
    nifti_img = nib.Nifti1Image(volume_data, np.eye(4))
    nib.save(nifti_img, nifti_path)  

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    print(img)
    return data

def show_3d_volume(data):
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    src = mlab.pipeline.scalar_field(data)
    vol_iso = mlab.pipeline.iso_surface(src, opacity=0.999, colormap='Greys')
    mlab.colorbar()
    mlab.view(azimuth=45, elevation=45, distance='auto')
    mlab.show()

import os

def delete_png_files(folder_path):
    try:
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                print(f"Deleted: {file}")
        print("Deletion complete.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    data_folder = "E:/Ned/Systems/rsna-miccai-brain-tumor-radiogenomic-classification/train"
    new_data_folder = "E:/Ned/Systems/datatraining"

    crop_size = (100, 100)  # Specify crop size
    target_size = (256, 256)  # Specify target size for resizing

    # Create a new folder for processed data
    os.makedirs(new_data_folder, exist_ok=True)

    # Process each folder in the original data folder
    for subject_folder in os.listdir(data_folder):
        subject_folder_path = os.path.join(data_folder, subject_folder)
        new_subject_folder_path = os.path.join(new_data_folder, subject_folder)

        # Create a new folder for the subject in the new data folder
        os.makedirs(new_subject_folder_path, exist_ok=True)

        # Process each modality folder within the subject folder
        for modality_folder in os.listdir(subject_folder_path):
            modality_folder_path = os.path.join(subject_folder_path, modality_folder)
            new_modality_folder_path = os.path.join(new_subject_folder_path, modality_folder)

            # Create a new modality folder in the new data folder
            os.makedirs(new_modality_folder_path, exist_ok=True)

            # Remove black images from the original data
            remove_black_images(modality_folder_path)

            # Convert DICOM to PNG
            convert_folder_to_png(modality_folder_path, new_modality_folder_path)

            # Crop and resize PNG images
            croppng(new_modality_folder_path, new_modality_folder_path)

            #resize_folder = os.path.join(new_modality_folder_path, "resize")
            #resize_images(new_modality_folder_path, resize_folder, size=target_size)

            # Convert PNG to NIfTI
            nifti_output_path = os.path.join(new_modality_folder_path, f"{modality_folder}.nii.gz")
            png_series_to_nifti(new_modality_folder_path, nifti_output_path)
            delete_png_files(new_modality_folder_path)
            # Load and visualize NIfTI data
#             try:
#                 nifti_data = load_nifti(nifti_output_path)
#                 # show_3d_volume(nifti_data)
#             except Exception as e:
#                 print(f"Error loading or visualizing NIfTI data: {e}")
