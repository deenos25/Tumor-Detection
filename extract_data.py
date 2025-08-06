import kagglehub
import cv2
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BrainTumorDataLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def _load_images_and_labels(self, set_type, med_type):
        image_dir = os.path.join(self.dataset_path, set_type, med_type, 'images')
        label_dir = os.path.join(self.dataset_path, set_type, med_type, 'labels')
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            logging.warning(f"Directories not found for set_type='{set_type}', med_type='{med_type}'. Skipping.")
            return [], [], []
        images_list = []
        classes_list = []
        box_coor_list = []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg'))]
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.txt'))]
        for filename in image_files:
            base_filename = os.path.splitext(filename)[0]
            label_filename = base_filename + '.txt'  # Adjust extension if needed
            if label_filename not in label_files:
                logging.warning(f"No corresponding label found for image {filename}. Skipping.")
                continue
            try:
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, label_filename)
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not read image file: {image_path}. Skipping.")
                    continue
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
                label = pd.read_csv(label_path, sep=' ', header=None).to_numpy(dtype=object)
                images_list.append(image_gray)
                classes_list.append(label[:, 0])
                box_coor_list.append(label[:, 1:])
            except Exception as e:
                logging.error(f"Error processing file {filename} or {label_filename}: {e}")
                continue
        return images_list, classes_list, box_coor_list

    def create_dataframe(self):
        data_rows = []
        set_types = ['Train', 'Val']
        med_types = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        for set_type in set_types:
            for med_type in med_types:
                images, classes, box_coor = self._load_images_and_labels(set_type=set_type, med_type=med_type)
                for i in range(len(images)):
                    data_rows.append({
                        'image': images[i],
                        'class': classes[i],
                        'box_coor': box_coor[i],
                        'set_type': set_type,
                        'med_type': med_type
                    })
        df = pd.DataFrame(data_rows)
        return df

    def save_data(self, name):
        logging.info("Starting data processing and DataFrame creation...")
        df = self.create_dataframe()
        logging.info(f"DataFrame created with {len(df)} rows. Saving to {name}...")
        df.to_pickle(name, compression='gzip')
        logging.info("Data saved successfully.")

path = kagglehub.dataset_download("ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes")
data_loader = BrainTumorDataLoader(path)
data_loader.save_data("C:\\Users\\deeno\\.cache\\processed_image_data.pkl.gz")