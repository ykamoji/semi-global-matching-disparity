import sys
import time as t
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        self.direction = direction
        self.name = name


# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')

paths = [N, NE, E, SE, S, SW, W, NW]
effective_paths = [(E, W), (SE, NW), (S, N), (SW, NE)]

blur_size = 3
window_size = 5
max_disparity = 55


def load_images(left_name, right_name):
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, (blur_size, blur_size), 0, 0)

    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, (blur_size, blur_size), 0, 0)

    return left, right


def get_indices(offset, dim, direction, height):
    y_indices = []
    x_indices = []

    for i in range(dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(i)
            else:
                y_indices.append(i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset):
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = 5
    penalties[np.abs(disparities - disparities.T) > 1] = 70

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume):
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1
    aggregation_volume = np.zeros(shape=(height, width, disparities, len(paths)), dtype=cost_volume.dtype)
    path_id = 0
    for path in effective_paths:
        print('\tProcessing paths {} and {}\t'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1), axis=0)

        if main.direction == E.direction:
            for y in range(height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2
        dusk = t.time()
        print('\t(Completed in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


def compute_costs(left, right):
    height = left.shape[0]
    width = left.shape[1]

    cheight = window_size
    cwidth = window_size

    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census', end='')
    sys.stdout.flush()
    dawn = t.time()
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = np.int64(0)
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_census_values[y, x] = left_census

            right_census = np.int64(0)
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_census_values[y, x] = right_census

    dusk = t.time()
    print(f'\t(Completed in {dusk - dawn:.2f}s)')

    print(f'\tComputing cost volumes\t', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, max_disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

    dusk = t.time()
    print('\t(Completed in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume


def select_disparity(aggregation_volume):
    volume = np.sum(aggregation_volume, axis=3)
    return np.argmin(volume, axis=2)


if __name__ == '__main__':
    read_path = "../data/disparity/"
    save_path = "../output/disparity/"
    images = [["teddy_im2.png", "teddy_im6.png"], ["cones_im2.png", "cones_im6.png"]]
    for left_name, right_name in images:
        dawn = t.time()

        left, right = load_images(read_path + left_name, read_path + right_name)

        print('\nStarting cost computation')
        cost_volume = compute_costs(left, right)

        print('\nStarting aggregation computation')
        aggregation_volume = aggregate_costs(cost_volume)

        print('\nSelecting best disparities')
        disparity_map = select_disparity(aggregation_volume)
        disparity_map = np.uint8(255.0 * disparity_map / max_disparity)

        disparity_map = cv2.medianBlur(disparity_map, blur_size)

        depth_map = 1 / (disparity_map + 1e-1)

        depth_cutoff = 0.02
        depth_map[depth_map > depth_cutoff] = 0

        depth_map = depth_map[window_size:-window_size, window_size + max_disparity:-window_size]

        plt.imshow(depth_map, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(save_path + left_name.split('_')[0] + "_sgm.png", bbox_inches='tight', edgecolor='auto')
        plt.show()

        dusk = t.time()
        print('\nCompleted !')
        print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))
