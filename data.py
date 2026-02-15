import os
import time
import cv2
import glob
import random
import sys

# --- 1. Import C++ Backend ---
try:
    from my_framework_cpp import Tensor
    print("✓ Successfully imported C++ Backend.")
except ImportError:
    print("! C++ Backend not found. Please compile it using: python setup.py build_ext --inplace")
    sys.exit(1)

# --- 2. The Dataset Class ---
class Dataset:
    def __init__(self, root_dir):
        """
        Initializes the dataset and automatically loads data.
        Constraint 3.2: Measure and report dataset loading time.
        """
        self.root_dir = root_dir
        self.samples = []  # Stores (Tensor, label_index)
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.loading_time = 0.0
        
        # Check if folder exists
        if not os.path.exists(root_dir):
            print(f"Error: Dataset path '{root_dir}' not found.")
            return

        print(f"Processing dataset: {root_dir} ...")
        
        # --- START TIMER ---
        start_time = time.time()
        
        self._load_data()
        
        # --- STOP TIMER ---
        end_time = time.time()
        self.loading_time = end_time - start_time
        
        # Constraint: Print loading time clearly
        print(f"✓ Done! Loading time: {self.loading_time:.4f} seconds")
        print(f"✓ Found {len(self.samples)} images across {len(self.class_to_idx)} classes.")

    def _load_data(self):
        print(f"Scanning {self.root_dir}...")
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory {self.root_dir} not found.")

        subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        subdirs.sort()
        
        total_loaded = 0
        
        for idx, class_name in enumerate(subdirs):
            self.class_to_idx[class_name] = idx
            class_path = os.path.join(self.root_dir, class_name)
            image_paths = glob.glob(os.path.join(class_path, "*.png"))
            
            # --- FAST MODE: SUBSET HACK ---
            # Keep this UN-COMMENTED for debugging (loads 100 images/class)
            # Comment it out later for the full 125MB run.
            image_paths = image_paths
            
            num_images = len(image_paths)
            print(f" -> Loading Class '{class_name}' ({num_images} images)...")
            
            for i, img_path in enumerate(image_paths):
                self._process_single_image(img_path, idx)
                total_loaded += 1
                
                # --- PROGRESS INDICATOR ---
                if (i + 1) % 500 == 0:
                    print(f"    Processed {i + 1}/{num_images} images...", end='\r')
            
            # print(f"    ✓ Done with '{class_name}'.") # Optional cleanup

        print(f"✓ FINISHED: Loaded {total_loaded} images total.")

    def _process_single_image(self, path, label):
        # 1. Load image using OpenCV
        img = cv2.imread(path)
        if img is None: return 

        # 2. Resize to 32x32
        img = cv2.resize(img, (32, 32))

        # 3. CONVERT TO TENSOR
        try:
            # Fast C++ Loader
            raw_bytes = img.tobytes() 
            # Note: OpenCV is BGR, usually we want RGB, but for this assignment consistency matters more
            t = Tensor.from_uint8(raw_bytes, [32, 32, 3])
            
        except AttributeError:
            # Fallback: Slow Python Loop
            flat_data = []
            height, width, channels = img.shape
            for h in range(height):
                for w in range(width):
                    b, g, r = img[h, w]
                    flat_data.append(float(r) / 255.0)
                    flat_data.append(float(g) / 255.0)
                    flat_data.append(float(b) / 255.0)
            t = Tensor(flat_data, [32, 32, 3])

        self.samples.append((t, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- 3. The DataLoader Class ---
class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
            
        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx : start_idx + self.batch_size]
            batch_tensors = []
            batch_labels = []
            
            for i in batch_indices:
                tensor, label = self.dataset[i]
                batch_tensors.append(tensor)
                batch_labels.append(label)
            yield batch_tensors, batch_labels

    def __len__(self):
        """Returns the number of batches."""
        import math
        return math.ceil(len(self.dataset) / self.batch_size)

# --- 4. MAIN BLOCK ---
if __name__ == "__main__":
    # Use raw string r"..." or forward slashes to avoid escape char issues
    dataset_folders = ["data_2/data_2"] 

    for folder_name in dataset_folders:
        print(f"\n{'='*40}")
        print(f"TESTING DATASET: {folder_name}")
        print(f"{'='*40}")

        if not os.path.exists(folder_name):
            print(f"Skipping {folder_name} (Folder not found)")
            continue

        dataset = Dataset(folder_name)