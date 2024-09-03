import numpy as np
from scipy.ndimage import convolve
from collections import Counter
from collections import deque



def filter_instances(instance_mask, labels_list, instance_list):
    filter_sparse_pixels_for_all_instances(instance_mask)
    no_way = way_to_boundaries(instance_mask)
    merge_instances(instance_mask, no_way)


    remaining_instances = set(instance_mask.flatten())


    for instance in instance_list[:]:
        if instance not in remaining_instances:
            index = instance_list.index(instance)
            del labels_list[index]
            del instance_list[index]



    #überprüfen ob noah alle Instanzen vorhanden sind

    filtered_labels_list = []
    filtered_instance_mask = np.zeros(instance_mask.shape)

    for i in range(len(labels_list)):
        class_of_instance = labels_list[i]
        if is_instance_in_foreground(instance_mask, instance_list[i]) or tiny_instance(instance_mask, instance_list[i]):
        #if is_instance_in_foreground(instance_mask, instance_list[i]) and not large_instance(instance_mask, instance_list[i]):
        #if is_train_or_bus_or_truck(class_of_instance) and is_instance_in_center_of_mask(instance_mask, instance_list[i]) and not large_instance(instance_mask, instance_list[i]):
            continue
        filtered_labels_list.append(class_of_instance)
        current_instance_mask = np.zeros(instance_mask.shape)
        current_instance_mask[instance_mask == instance_list[i]] = len(filtered_labels_list)
        filtered_instance_mask = np.add(filtered_instance_mask, current_instance_mask)
    filtered_instance_list = list(range(1, len(filtered_labels_list) + 1))
    return filtered_instance_mask, filtered_labels_list, filtered_instance_list






def is_instance_in_center_of_mask(mask, instance_id):
    height, width = mask.shape

    # Definiere den mittleren Drittel-Bereich
    mid_start_y, mid_end_y = height // 3, 2 * height // 3
    mid_start_x, mid_end_x = width // 3, 2 * width // 3

    # Extrahiere das mittlere Drittel der Maske
    central_region = mask[mid_start_y:mid_end_y, mid_start_x:mid_end_x]

    # Überprüfen, ob die Instanz-ID im mittleren Drittel vorhanden ist
    return np.any(central_region == instance_id)


def is_train_or_bus_or_truck(class_of_instance):
    return class_of_instance in [20, 21, 24]

def large_instance(instance_mask, instance_id):
    return np.sum(instance_mask == instance_id) > 1000

def tiny_instance(instance_mask, instance_id):
    return np.sum(instance_mask == instance_id) < 10

def is_instance_in_foreground(mask, instance_id):
    height, width = mask.shape

    # Define the foreground region (last third of height)
    fg_start_y, fg_end_y = 2 * height // 3, height

    # Extract the foreground region of the mask (entire width in the last third of height)
    foreground_region = mask[fg_start_y:fg_end_y, :]

    # Count the total number of pixels belonging to the instance in the entire mask
    total_instance_pixels = np.sum(mask == instance_id)

    # Count the number of pixels belonging to the instance in the foreground region
    foreground_instance_pixels = np.sum(foreground_region == instance_id)

    # Calculate the percentage of the instance in the foreground region
    foreground_percentage = foreground_instance_pixels / total_instance_pixels if total_instance_pixels > 0 else 0

    # Check if 70% or more of the instance is in the foreground region
    return foreground_percentage >= 0.7






def high_density(mask, instance_id, density_threshold=0.2):


    # Get the indices where the instance_id is located in the mask
    instance_indices = np.where(mask == instance_id)



    # Determine the bounding box of the instance
    min_y, max_y = instance_indices[0].min(), instance_indices[0].max()
    min_x, max_x = instance_indices[1].min(), instance_indices[1].max()

    # Extract the region within the bounding box
    region = mask[min_y:max_y + 1, min_x:max_x + 1]

    # Count the number of instance ID pixels and the number of zero pixels
    num_instance_pixels = np.sum(region == instance_id)
    num_zero_pixels = np.sum(region == 0)

    # Calculate the density of instance ID pixels within the region
    density = num_instance_pixels / (num_instance_pixels + num_zero_pixels)

    # Check if the density is above the threshold
    if density >= density_threshold:
        return True
    return False


def filter_sparse_pixels_for_all_instances(instance_mask, neighborhood_size=5, threshold=5):
    """
    Sets pixels in the instance mask to 0 if their neighborhood has few pixels of the same instance.

    Args:
        instance_mask (numpy.ndarray): The input mask with instance IDs.
        neighborhood_size (int): The size of the neighborhood to consider (must be odd).
        threshold (int): The minimum number of same instance pixels required in the neighborhood.

    Returns:
        numpy.ndarray: The filtered instance mask.
    """


    # Get all unique instance IDs (excluding 0, which is the background)
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude background (assumed to be 0)

    # Define the neighborhood filter
    neighborhood_filter = np.ones((neighborhood_size, neighborhood_size))

    # Iterate over all unique instance IDs
    for instance_id in instance_ids:
        # Create a binary mask for the specific instance
        binary_mask = (instance_mask == instance_id).astype(int)

        # Convolve the binary mask with the neighborhood filter
        neighbor_count = convolve(binary_mask, neighborhood_filter, mode='constant', cval=0)

        # Identify sparse pixels where the number of same-instance neighbors is below the threshold
        sparse_pixels = (neighbor_count < threshold) & (instance_mask == instance_id)

        # Set these sparse pixels to 0 in the filtered mask
        instance_mask[sparse_pixels] = 0





def merge_instances(mask, no_way):
    width, height = mask.shape

    for x in range(width):
        for y in range(height):
            if not caged(mask, x, y, no_way):
                continue
            neighbours = get_all_neighbours(mask, x, y)
            non_zero_neighbours = [n for n in neighbours if n[2] != 0 and n[2] != mask[x, y]]
            if len(non_zero_neighbours) == 0:
                continue
            most_frequent_value = most_frequent([n[2] for n in non_zero_neighbours])
            mask[x, y] = most_frequent_value


def get_all_neighbours(mask, x, y):
    neighbours = []
    width, height = mask.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        if 0 <= nx < width and 0 <= ny < height:
            neighbours.append((nx, ny, mask[nx, ny]))

    return neighbours

def get_neighbours(mask, x, y):
    neighbours = []
    width, height = mask.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy


        if 0 <= nx < width and 0 <= ny < height:
            neighbours.append((nx, ny, mask[nx, ny]))

    return neighbours

def most_frequent(values):
    return Counter(values).most_common(1)[0][0]



def caged(mask, x, y, no_way):
    instance = mask[x, y]

    if instance == 0:
        return (x, y) in no_way


    queue = deque([(x, y)])
    visited = {(x, y)}

    while queue:
        cx, cy = queue.popleft()
        neighbours = get_neighbours_coords(mask, cx, cy)

        for nx, ny in neighbours:
            if (nx, ny) not in visited:
                if mask[nx, ny] == 0:
                    return False

                elif mask[nx, ny] == instance:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return True

def get_neighbours_coords(mask, x, y):
    width, height = mask.shape
    neighbours = []

    # Nachbarpositionen (rechts, links, oben, unten)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        # Nur gültige Nachbarn hinzufügen
        if 0 <= nx < width and 0 <= ny < height:
            neighbours.append((nx, ny))

    return neighbours


def bfs_find_way(mask, x, y, way, no_way):
    queue = deque([(x, y)])
    visited = {(x, y)}

    width, height = mask.shape


    while queue:
        cx, cy = queue.popleft()

        # Überprüfe, ob wir den Rand erreicht haben
        if cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
            way.extend(visited)
            return True

        for nx, ny in get_neighbours_coords(mask, cx, cy):
            if mask[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))


    no_way.extend(visited)
    return False


def way_to_boundaries(mask):
    no_way = []
    way = []
    visited_global = set()

    width, height = mask.shape
    for x in range(width):
        for y in range(height):
            if mask[x, y] == 0 and (x, y) not in visited_global:
                if bfs_find_way(mask, x, y, way, no_way):
                    visited_global.update(way)
                else:
                    visited_global.update(no_way)

    return no_way



