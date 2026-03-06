#Now skeleton is correctly showing actual edges
#can replace v5&6
#Output total pixels

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, remove_small_holes, label
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from skan import Skeleton, summarize

# =================CONFIG=================
INPUT_DIR = r"C:\Work\Exp\E1\pre\HP326v61\cleanup\topo1"
OUTPUT_DIR = r"C:\Work\Exp\E1\pre\HP326v61\cleanup\topo1\lccglob_eff"
# ========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_largest_connected_component(binary_image):
    # Label connected components
    labeled_image = label(binary_image, connectivity=2)
    
    # Count pixels in each component (excluding background=0)
    component_sizes = np.bincount(labeled_image.ravel())
    component_sizes[0] = 0  # Ignore background
    
    if len(component_sizes) <= 1:
        return binary_image
    
    # Find largest component
    largest_component_label = component_sizes.argmax()
    lcc_image = (labeled_image == largest_component_label)
    
    print(f"  Found {len(component_sizes)-1} connected components")
    print(f"  Largest component has {component_sizes[largest_component_label]} pixels")
    
    return lcc_image

def calculate_efficiencies(G, node_coords_list):
    N = len(node_coords_list)
    if N < 2:
        return 0.0, 0.0

    coords_arr = np.array(node_coords_list)
    dE_matrix = squareform(pdist(coords_arr, 'euclidean'))
    nodelist = list(range(N))
    dG_matrix = nx.floyd_warshall_numpy(G, nodelist=nodelist, weight='weight')
    dT_matrix = nx.floyd_warshall_numpy(G, nodelist=nodelist, weight=None)

    sum_eg = 0.0
    sum_et = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dE = dE_matrix[i, j]
            dg = dG_matrix[i, j]
            sum_eg += (dE / dg) if (not np.isinf(dg) and dg != 0) else 0
            dt = dT_matrix[i, j]
            sum_et += (dE / dt) if (not np.isinf(dt) and dt != 0) else 0

    factor = 2.0 / (N * (N - 1))
    return sum_eg * factor, sum_et * factor

def visualize_and_save(img_shape, G, node_coords_map, title, output_path, skel_img=None):
    plt.figure(figsize=(12, 12))
    if skel_img is not None:
        rgb_img = np.zeros((*img_shape, 3), dtype=np.uint8)
        rgb_img[skel_img, :] = [255, 255, 0]  # Yellow
        plt.imshow(rgb_img)
    else:
        plt.imshow(np.zeros(img_shape, dtype=np.uint8), cmap='gray')
        nx.draw_networkx_edges(G, node_coords_map, edge_color='yellow', width=1, alpha=0.7)
    
    pos = node_coords_map
    nx.draw_networkx_nodes(G, pos, node_size=1, node_color='red', edgecolors='red', linewidths=0.5)
    plt.title(title, fontsize=16, color='white', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()

def process_image(img_path, filename, output_sub_dir):
    print(f"Processing: {filename}")
    img_raw = imread(img_path)
    if img_raw.ndim == 3 and img_raw.shape[-1] == 4:
        img_gray = rgb2gray(img_raw[..., :3])
    elif img_raw.ndim == 3:
        img_gray = rgb2gray(img_raw)
    else:
        img_gray = img_raw

    binary = img_gray > 0.05
    binary_lcc = extract_largest_connected_component(binary)
    binary_lcc = remove_small_holes(binary_lcc, max_size=2500)
    skel_img = skeletonize(binary_lcc)

    skel_obj = Skeleton(skel_img)
    branch_data = summarize(skel_obj, separator='-')
    
    src = branch_data[['node-id-src', 'image-coord-src-0', 'image-coord-src-1']].rename(
        columns={'node-id-src':'id','image-coord-src-0':'r','image-coord-src-1':'c'})
    dst = branch_data[['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1']].rename(
        columns={'node-id-dst':'id','image-coord-dst-0':'r','image-coord-dst-1':'c'})
    nodes_df = pd.concat([src, dst]).drop_duplicates(subset='id').set_index('id')
    
    skan_ids = nodes_df.index.values
    N = len(skan_ids)
    if N < 2: return None

    id_map = {sid: i for i, sid in enumerate(skan_ids)}
    node_coords = [(nodes_df.loc[sid, 'c'], nodes_df.loc[sid, 'r']) for sid in skan_ids]
    node_map = {i: coords for i, coords in enumerate(node_coords)}

    G_skel = nx.Graph()
    for _, row in branch_data.iterrows():
        G_skel.add_edge(id_map[row['node-id-src']], id_map[row['node-id-dst']], weight=row['branch-distance'])

    coords_arr = np.array(node_coords)
    dE_mat = squareform(pdist(coords_arr, 'euclidean'))
    G_comp = nx.Graph()
    for i in range(N):
        for j in range(i+1, N): G_comp.add_edge(i, j, weight=dE_mat[i,j])
    G_mst = nx.minimum_spanning_tree(G_comp, weight='weight')
    
    G_del = nx.Graph()
    if N >= 3:
        for simplex in Delaunay(coords_arr).simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                if not G_del.has_edge(u, v): G_del.add_edge(u, v, weight=dE_mat[u,v])
    else:
        G_del.add_edge(0, 1, weight=dE_mat[0,1])

    # Calculate Total Pixels (Edge Weights Sum)
    total_pix_skel = G_skel.size(weight='weight')
    total_pix_mst = G_mst.size(weight='weight')
    total_pix_del = G_del.size(weight='weight')

    base = os.path.splitext(filename)[0]
    visualize_and_save(skel_img.shape, G_skel, node_map, "Skeleton (LCC)", os.path.join(output_sub_dir, f"{base}_skel_lcc.png"), skel_img=skel_img)
    visualize_and_save(skel_img.shape, G_mst, node_map, "MST (LCC)", os.path.join(output_sub_dir, f"{base}_mst_lcc.png"))
    visualize_and_save(skel_img.shape, G_del, node_map, "Delaunay (LCC)", os.path.join(output_sub_dir, f"{base}_del_lcc.png"))

    eg_skel, et_skel = calculate_efficiencies(G_skel, node_coords)
    eg_mst, et_mst = calculate_efficiencies(G_mst, node_coords)
    eg_del, et_del = calculate_efficiencies(G_del, node_coords)

    return {
        'Image': filename, 'Nodes': N,
        'Eg_Skel': eg_skel, 'Et_Skel': et_skel, 'Total_Pix_Skel': total_pix_skel,
        'Eg_MST': eg_mst, 'Et_MST': et_mst, 'Total_Pix_MST': total_pix_mst,
        'Eg_Del': eg_del, 'Et_Del': et_del, 'Total_Pix_Del': total_pix_del
    }

def main():
    ensure_dir(OUTPUT_DIR)
    files = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    results = []
    for f in files:
        res = process_image(f, os.path.basename(f), OUTPUT_DIR)
        if res: results.append(res)
    if results:
        pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "global_efficiency_results_lcc.csv"), index=False)

if __name__ == "__main__":
    main()