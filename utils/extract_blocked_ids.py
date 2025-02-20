import os

def get_excluded_class_indices(train_folder_path, excluded_classes):
    folder_names = os.listdir(train_folder_path)  # List all folders in the train directory
    excluded_indices = []

    for index, folder_name in enumerate(folder_names):
        if folder_name in excluded_classes:
            excluded_indices.append(index)
    
    return excluded_indices

# Example usage
train_folder_path = "/home/rechenschieber/Documents/GitHub/embed-then-classify/data/train"  # Replace with the actual train folder path
EXCLUDED_CLASSES = {
    "n04201297", "n04204347", "n04239074", "n04277352", "n04370456", "n04409515",
    "n04456115", "n04479046", "n044873942", "n04525038", "n04591713", "n04599235", 
    "n07565083", "n07613480", "n07695742", "n07714571", "n07717410", "n07753275",
    "n10148035", "n12768682"
}

excluded_indices = get_excluded_class_indices(train_folder_path, EXCLUDED_CLASSES)
print(excluded_indices)
