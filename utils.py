def extract_edge(arr):
    """
    Extracts a 1-pixel-width edge from a binarized ndarray 
    (e.g. an ice chart with 1 representing sea ice and 0 representing open water)
    """
    from scipy import ndimage

    return (ndimage.morphology.binary_dilation(arr) - arr)


def compute_edge_length(ic_edge, pixel_spacing=1):
    """
    Computes the length of an edge in an ndarray. 

    ic_edge is an ndarray with a 1-pixel width edge - with edge pixels being assigned a value of 1,
    while all other pixels are zeroed out.
    
    Provide a pixel_spacing to get the length in a real unit rather than pixels.

    Reference: https://os.copernicus.org/articles/15/615/2019/

    TODO: does not handle pixels at the very edge correctly - the discrepancy is negligible, though (less than a pixel).
    """
    import numpy as np

    ic_edge_coords = np.argwhere(ic_edge == 1)
    all_xs = ic_edge_coords[:, 0]
    all_ys = ic_edge_coords[:, 1]

    diagonal_cells = 0
    non_diagonal_cells = 0
    mix_cells = 0
    for ic_edge_coord in ic_edge_coords:
        n_neighbors = 0
        x = ic_edge_coord[0]
        y = ic_edge_coord[1]
                        
        x_idxs = np.argwhere(all_xs == x)
        ys = all_ys[x_idxs]
        if y + 1 in ys:
            n_neighbors +=1
        if y - 1 in ys:
            n_neighbors +=1
            
        x_idxs = np.argwhere(all_xs == x + 1)
        ys = all_ys[x_idxs]
        if y in ys:
            n_neighbors +=1
            
        x_idxs = np.argwhere(all_xs == x - 1)
        ys = all_ys[x_idxs]
        if y in ys:
            n_neighbors +=1
            
        if n_neighbors >= 2:
            non_diagonal_cells +=1
        if n_neighbors == 1:
            mix_cells += 1
        if n_neighbors == 0:
            diagonal_cells += 1

    return non_diagonal_cells*pixel_spacing + diagonal_cells*np.sqrt(2)*pixel_spacing + mix_cells*0.5*(pixel_spacing + np.sqrt(2)*pixel_spacing)


def compute_IIEE_area(chart1, chart2, pixel_spacing=1):
    """
    The function returns the IIEE area for two binarized ice charts on a common grid. 

    Provide a pixel_spacing to get the length in a real unit rather than pixels.

    Reference: https://os.copernicus.org/articles/15/615/2019/
    """
    import numpy as np

    sym_diff = np.zeros(chart1.shape)
    sym_diff[(chart1 == 1) & (chart2 == 0)] = 1
    sym_diff[(chart2 == 0) & (chart1 == 1)] = 1

    return (sym_diff == 1).sum()*pixel_spacing**2


def compute_avg_displacement(edge1, edge2, pixel_spacing=1):
    """
    Computes the average displacement between two edges.
    """
    import numpy as np
    
    edge1_coords = np.argwhere(edge1 == 1)
    edge2_coords = np.argwhere(edge2 == 1)

    edge1_deviations = []
    for pixel in edge1_coords:
        distances = np.sqrt((edge2_coords[:, 0]-pixel[0])**2 + (edge2_coords[:, 1]-pixel[1])**2)
        edge1_deviations.append(distances[np.argmin(distances)])

    edge2_deviations = []
    for pixel in edge2_coords:
        distances = np.sqrt((edge1_coords[:, 0]-pixel[0])**2 + (edge1_coords[:, 1]-pixel[1])**2)
        edge2_deviations.append(distances[np.argmin(distances)])

    return pixel_spacing*(np.mean(edge1_deviations) + np.mean(edge2_deviations))/2