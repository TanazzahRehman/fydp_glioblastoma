from flask import Flask, render_template, request, jsonify, send_file
import os
from inference import inference, load_model, unzip_dicom, remove_black_images, convert_folder_to_png, croppng, \
    png_series_to_nifti, delete_png_files

app = Flask(__name__)

model_path = "best_metric_model.pth"  # Path to the trained model
model = load_model(model_path)
@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/visualize.html')
def visualize():
    return render_template('visualize.html')

@app.route('/faqs.html')
def faqs():
    return render_template('faqs.html')

@app.route('/check_directory')
def check_directory():
    try:
        dir_path = './processed_data'
        print(f"Checking directory: {dir_path}")
        
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"Found files: {files}")
            return jsonify({'files': files})
        else:
            print("Directory not found")
            return jsonify({'files': []})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/processed_data/output.nii.gz')
def serve_outputnifti():
    nifti_path = './processed_data/output.nii.gz'
    try:
        return send_file(nifti_path, as_attachment=False)
    except Exception as e:
        print(f"Error serving NIfTI file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/spmSmall.nii.gz')
def serve_spmSmallnifti():
    nifti_path = 'spmSmall.nii.gz'
    try:
        return send_file(nifti_path, as_attachment=False)
    except Exception as e:
        print(f"Error serving NIfTI file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Save the uploaded zip file
        zip_path = "uploads/" + file.filename
        file.save(zip_path)

        # Extract the zip file
        extract_to = "extracted_data"
        unzip_dicom(zip_path, extract_to)

        # Create a new folder for processed data
        new_data_folder = "processed_data"
        os.makedirs(new_data_folder, exist_ok=True)

        # Path to the folder containing DICOM files
        data_folder = os.path.join(extract_to, os.path.splitext(file.filename)[0])

        # Remove black images from the original data
        remove_black_images(data_folder)

        # Convert DICOM to PNG
        convert_folder_to_png(data_folder, new_data_folder)

        # Crop and resize PNG images
        croppng(new_data_folder, new_data_folder)

        # Convert PNG to NIfTI
        nifti_output_path = os.path.join(new_data_folder, "output.nii.gz")
        png_series_to_nifti(new_data_folder, nifti_output_path)

        # Delete PNG files after conversion to NIfTI
        delete_png_files(new_data_folder)
        prediction = inference(model,nifti_output_path)
        # Clean up extracted data
        os.remove(zip_path)

        return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
