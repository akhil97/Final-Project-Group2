import os
import pandas as pd
import random

image_dataset_dir = '/home/ubuntu/DL/Project/Data/'

# Set the target categories
categories = ['coast', 'coast_ship', 'detail', 'land', 'multi', 'ship', 'water'] # Replace with your actual categories

# Set the split ratios for train, validation, and test sets
train_split_ratio = 0.8
val_split_ratio = 0.10
test_split_ratio = 0.10

# Create an empty list to store the image file paths and their corresponding labels
data = []

# Iterate through the categories and their respective directories
for category in categories:
    category_dir = os.path.join(image_dataset_dir, category)

    # Get a list of image file names in the category directory
    image_names = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

    # Randomly shuffle the list of image file names
    random.shuffle(image_names)

    # Calculate the number of images for each split
    num_train_images = int(len(image_names) * train_split_ratio)
    num_val_images = int(len(image_names) * val_split_ratio)
    num_test_images = int(len(image_names) * test_split_ratio)

    # Assign the split for each image file and add the file path and label to the data list
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(category_dir, image_name)
        if i < num_train_images:
            split = 'train'
        elif i < num_train_images + num_val_images:
            split = 'val'
        else:
            split = 'test'
        data.append([image_name, category, split, category])


# Convert the data list to a DataFrame
df = pd.DataFrame(data, columns=['image_name', 'target', 'split', 'one_hot'])

# Perform one-hot encoding on the 'class' column
one_hot_encoded = pd.get_dummies(df['one_hot'])
one_hot_encoded = one_hot_encoded.astype(str) # Convert the one-hot encoded values to strings
df['target_class'] = one_hot_encoded.apply(lambda x: ','.join(x), axis=1)
df.drop('one_hot', axis=1, inplace=True)



# Save the DataFrame to an Excel file in the destination directory
excel_file_path = os.path.join("/home/ubuntu/DL/Project/Excel/", 'image_dataset.xlsx')
df.to_excel(excel_file_path, index=False)
