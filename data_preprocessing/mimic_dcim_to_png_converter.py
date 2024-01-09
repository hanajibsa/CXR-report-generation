import os
import numpy as np
from PIL import Image
import pydicom as dicom

directory = "/mnt/gpfs2_4m/scratch/sgu260/mimic-cxr-2/physionet.org/files/mimic-cxr/2.0.0/files/p11/"

list_dir = os.listdir(directory)

for x in list_dir:
    path = directory + x
    if os.path.isdir(path):
        files = os.listdir(path)
        for y in files:
            path2 = path + "/" + y
            if os.path.isdir(path2):
                files2 = os.listdir(path2)
                for z in files2:
                    if z[-3:] == "dcm":
                        image_path = path2 + "/" + z
                        ds = dicom.dcmread(image_path)

                        try:
                            pixel_array = ds.pixel_array.astype(float)
                            scaled_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
                            png_array = scaled_array.astype(np.uint8)

                            # Convert the PNG array to a PIL Image object
                            pil_image = Image.fromarray(png_array)
                            image_path = image_path[:-3] + "png"
                            # Save the PIL Image as a PNG file
                            pil_image.save(image_path)
                            print("Successfully save png to: ", image_path)
                        
                        except:
                            print("Some error occured saving png: ", image_path)
                            continue
                            