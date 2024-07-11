import os
import zipfile
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
import cv2
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityD, EnsureTyped, ResizeD
from torch.utils.data import DataLoader


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet = resnet18(spatial_dims=3, n_input_channels=1, shortcut_type='A', bias_downsample=False,
                               feed_forward=False, pretrained=True)
        # Check if self.resnet.fc exists, and if not, create one
        if hasattr(self.resnet, 'fc') and self.resnet.fc is not None:
            in_features = self.resnet.fc.in_features
        else:
            in_features = 512  # Default size if fc is not found, might need adjustment based on the actual architecture

        self.resnet.fc = nn.Linear(in_features, 1)  # Adjust the final layer to output a single value

    def forward(self, x):
        x = self.resnet(x)
        return x


# Define the transforms for inference
# Define the transforms for inference
inference_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    ScaleIntensityD(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(64, 64, 64)),
    EnsureTyped(keys=["image"])
])


# Define a function to load the model
def load_model(model_path, device):
    model = CustomResNet18()
    model.to(device)  # Move model to the device
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.eval()
    return model


# Define a function for inference
def inference(model, image_path):
    image_data = {'image': image_path}
    transformed_data = inference_transforms(image_data)
    image = transformed_data['image'].unsqueeze(0)  # Add batch dimension

    # Move image to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Debugging prints
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Image is on device: {image.device}")

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).cpu().numpy()
        # prediction = 1 if prediction > 0.5 else 0  # Convert to binary prediction
        print(prediction)
    return prediction

    # Perform inference
    prediction = inference(model, nii_file_path)
    print(f"Prediction: {prediction}")


def unzip_dicom(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted zip file to: {extract_to}")


def remove_black_images(folder_path):
    removed_files = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, file))
                if np.max(ds.pixel_array) == 0:
                    os.remove(os.path.join(root, file))
                    print(f"Removed black image: {os.path.join(root, file)}")
                    removed_files += 1
    if removed_files == 0:
        print("No black images found to remove.")


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
    if not dicom_files:
        print(f"No DICOM files found in {input_folder}")
        return

    for dicom_file in dicom_files:
        dicom_path = os.path.join(input_folder, dicom_file)
        dicom_to_png(dicom_path, output_folder)


def croppng(folderinput, folderoutput):
    minx = 600
    miny = 600
    maxx = 0
    maxy = 0
    files = [file for file in os.listdir(folderinput) if file.endswith('.png')]

    if not files:
        print(f"No PNG files found in {folderinput}")
        return

    for i in files:
        name = os.path.join(folderinput, i)
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape
        thresh[hh - 3:hh, 0:ww] = 0
        white = np.where(thresh == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        if xmin < minx:
            minx = xmin
        if xmax > maxx:
            maxx = xmax
        if ymin < miny:
            miny = ymin
        if ymax > maxy:
            maxy = ymax
        print(f"File {i} - xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

    for j in files:
        name = os.path.join(folderinput, j)
        img = cv2.imread(name)
        crop = img[miny:maxy + 3, minx:maxx]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(crop)
        img_resized = pil_image.resize((64, 64), Image.LANCZOS)
        savecrop = os.path.join(folderoutput, j)
        img_resized_np = np.array(img_resized)
        cv2.imwrite(savecrop, img_resized_np)
        print(f"Cropped and resized image saved at: {savecrop}")


def png_series_to_nifti(png_folder, nifti_path):
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith(".png")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

    if not png_files:
        print(f"No PNG files found in {png_folder}")
        return

    first_image_path = os.path.join(png_folder, png_files[0])
    first_image = Image.open(first_image_path)
    img_shape = (len(png_files), first_image.height, first_image.width)

    volume_data = np.zeros(img_shape, dtype=np.uint8)
    for i, png_file in enumerate(png_files):
        image_path = os.path.join(png_folder, png_file)
        img = Image.open(image_path)
        volume_data[i, :, :] = np.array(img)
    nifti_img = nib.Nifti1Image(volume_data, np.eye(4))
    nib.save(nifti_img, nifti_path)
    print(f"NIfTI file saved at: {nifti_path}")


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


import os
import zipfile
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
import cv2
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstD, ScaleIntensityD, EnsureTyped, ResizeD
from torch.utils.data import DataLoader


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet = resnet18(spatial_dims=3, n_input_channels=1, shortcut_type='A', bias_downsample=False,
                               feed_forward=False, pretrained=True)
        # Check if self.resnet.fc exists, and if not, create one
        if hasattr(self.resnet, 'fc') and self.resnet.fc is not None:
            in_features = self.resnet.fc.in_features
        else:
            in_features = 512  # Default size if fc is not found, might need adjustment based on the actual architecture

        self.resnet.fc = nn.Linear(in_features, 1)  # Adjust the final layer to output a single value

    def forward(self, x):
        x = self.resnet(x)
        return x


# Define the transforms for inference
# Define the transforms for inference
inference_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    ScaleIntensityD(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(64, 64, 64)),
    EnsureTyped(keys=["image"])
])


# Define a function to load the model
def load_model(model_path, device):
    model = CustomResNet18()
    model.to(device)  # Move model to the device
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.eval()
    return model


# Define a function for inference
def inference(model, image_path):
    image_data = {'image': image_path}
    transformed_data = inference_transforms(image_data)
    image = transformed_data['image'].unsqueeze(0)  # Add batch dimension

    # Move image to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Debugging prints
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Image is on device: {image.device}")

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).cpu().numpy()
        # prediction = 1 if prediction > 0.5 else 0  # Convert to binary prediction
        print(prediction)
    return prediction

    # Perform inference
    prediction = inference(model, nii_file_path)
    print(f"Prediction: {prediction}")


def unzip_dicom(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted zip file to: {extract_to}")


def remove_black_images(folder_path):
    removed_files = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, file))
                if np.max(ds.pixel_array) == 0:
                    os.remove(os.path.join(root, file))
                    print(f"Removed black image: {os.path.join(root, file)}")
                    removed_files += 1
    if removed_files == 0:
        print("No black images found to remove.")


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
    if not dicom_files:
        print(f"No DICOM files found in {input_folder}")
        return

    for dicom_file in dicom_files:
        dicom_path = os.path.join(input_folder, dicom_file)
        dicom_to_png(dicom_path, output_folder)


def croppng(folderinput, folderoutput):
    minx = 600
    miny = 600
    maxx = 0
    maxy = 0
    files = [file for file in os.listdir(folderinput) if file.endswith('.png')]

    if not files:
        print(f"No PNG files found in {folderinput}")
        return

    for i in files:
        name = os.path.join(folderinput, i)
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape
        thresh[hh - 3:hh, 0:ww] = 0
        white = np.where(thresh == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        if xmin < minx:
            minx = xmin
        if xmax > maxx:
            maxx = xmax
        if ymin < miny:
            miny = ymin
        if ymax > maxy:
            maxy = ymax
        print(f"File {i} - xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

    for j in files:
        name = os.path.join(folderinput, j)
        img = cv2.imread(name)
        crop = img[miny:maxy + 3, minx:maxx]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(crop)
        img_resized = pil_image.resize((64, 64), Image.LANCZOS)
        savecrop = os.path.join(folderoutput, j)
        img_resized_np = np.array(img_resized)
        cv2.imwrite(savecrop, img_resized_np)
        print(f"Cropped and resized image saved at: {savecrop}")


def png_series_to_nifti(png_folder, nifti_path):
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith(".png")],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))

    if not png_files:
        print(f"No PNG files found in {png_folder}")
        return

    first_image_path = os.path.join(png_folder, png_files[0])
    first_image = Image.open(first_image_path)
    img_shape = (len(png_files), first_image.height, first_image.width)

    volume_data = np.zeros(img_shape, dtype=np.uint8)
    for i, png_file in enumerate(png_files):
        image_path = os.path.join(png_folder, png_file)
        img = Image.open(image_path)
        volume_data[i, :, :] = np.array(img)
    nifti_img = nib.Nifti1Image(volume_data, np.eye(4))
    nib.save(nifti_img, nifti_path)
    print(f"NIfTI file saved at: {nifti_path}")


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
