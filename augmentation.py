import os
from helper import save_dataset, create_dir
from data import load_data

#path = "/content/drive/MyDrive/Master/polypsIdiot/PNG"
path = "PNG"

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")
print(f"Test : {len(test_x)} - {len(test_y)}")

#Creating the folders

#directory in which will be saved augmented dataset
save_dir = os.path.join("dataset", "aug")
for item in ["train", "valid", "test"]:
  create_dir(os.path.join(save_dir, item, "images"))
  create_dir(os.path.join(save_dir, item, "masks"))


save_dataset(train_x, train_y, os.path.join(save_dir, "train"), augment=True)
save_dataset(valid_x, valid_y, os.path.join(save_dir, "valid"), augment=False)

save_dataset(test_x, test_y, os.path.join(save_dir, "test"), augment=False)
