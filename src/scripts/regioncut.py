import numpy as np
import numba as nb
import plotly.graph_objects as go

@nb.njit(parallel=True, fastmath=True, cache=True)
def momentum_regions(pos):
    L_1 = 220* (pos-50) + 10000
    L_2 = 220*(pos-50) + 25000
    return L_1, L_2

@nb.njit(parallel=True, fastmath=True, cache=True)


def get_highL(halo, low_L):
    ids = []
    # print("lista ids:", ids)
    # print(halo[:,8][low_L].shape[0])
    for i in range(halo[:,8][low_L].shape[0]):
        l1, l2 = momentum_regions(halo[:,8][low_L][i])
        # print(l1, l2)
        if np.where((halo[:,10][low_L][i] > l1) & (halo[:,10][low_L][i] < l2) & (halo[:,8][low_L][i] > 50)):
            ids.append(halo[:,7][low_L][i])
    print("longitud lista ids:", len(ids))
    return ids
def regions(pos):
    L_1 = 220*(pos-50) + 10000
    L_2 = 220*(pos-50) + 25000
    return L_1, L_2

# def find_ids(mag_dict, low_l):
#     prefilter = np.asarray(list(mag_dict.items()))
#
#     ids = []
#     for i in range (len(prefilter[low_l])):
#         line1, line2 = regions(prefilter[i][1][0])
#         if (prefilter[i][1][1] > line1 ) & (prefilter[i][1][1] < line2) & (prefilter[i][1][0] > 50):
#             ids.append(prefilter[i][0])
#     return ids
#     print("hh",prefilter)
#


    # lindex_L_reg = nb.typed.List.empty_list(nb.types.int64)
    # for i in nb.prange(pos_mag_lmc[low_L].shape[0]):
    #     line1, line2 = momentum_regions(pos_mag_lmc[low_L][i])
    #     if (L_mag_mw_lmc[low_L][i] > line1 ) & (L_mag_mw_lmc[low_L][i] < line2) & (pos_mag_lmc[low_L][i] > 50):
    #         lindex_L_reg.append(i)
    # return lindex_L_reg

# def add_id(array, id):
#     arr = np.column_stack((array, id))
#     return arr

def L_delta(pert, no_pert, threshold):
    Lx = pert[:,11] - no_pert[:,11]
    Ly = pert[:,12] - no_pert[:,12]
    Lz = pert[:,13] - no_pert[:,13]
    L_mag_delta = pert[:,10] - no_pert[:,10]
    sort_L = -np.sort(-L_mag_delta)
    percent = 100 * (L_mag_delta / no_pert[:,10])
    # print(f'percent shape {percent.shape}')
    # print(f'first 10 percent {percent[:10]}')
    id_threshold = np.where(percent > threshold)
    ids = no_pert[:, 7][id_threshold]
    ids = ids[:, np.newaxis]
    # print(f'ids shape {ids.shape}')
    #Revisar esto
    ids_in_pert = np.isin(pert[:, 7],ids)
    # print(f'first 10 ids in pert {ids_in_pert[30:100]}')
    # print(f'ids in pert shape {ids_in_pert.shape}')
    sel_pert = np.where(ids_in_pert == True)
    sel_3 = pert[sel_pert]
    # print(f'first 10 sel 3 {sel_3[:10]}')


    # print(f'id_threshold {threshold}')
    # np.isin(id_threshold, L_mag_delta)
    # above_threshold = pert[id_threshold]
    # print("the shape of above threshold", above_threshold.shape)
    # L_mag_delta = np.linalg.norm(np.array([Lx, Ly, Lz]), axis=0)
    return sel_3

# def L_mag_delta(L_mag_1, L_mag_2):
#     L_mag_delta = L_mag_2 - L_mag_1
#     return L_mag_delta
# def idx_percentage(L_mag_delta, L_mag, threshold):
#
#     return idx

def threshold_L(halo,initial, threshold):
    # here we select the particles that are below the threshold
    low_L = np.where( (halo[:, 10] < threshold) & (halo[:, 10] > initial))
    ids_nopert = halo[:, 7][low_L]
    print("the shape of pos below threshold", halo[:, 10][low_L].shape)
    print("the shape of ids below threshold", ids_nopert.shape)
    print("first 10 ids", ids_nopert[:10])
    return low_L, ids_nopert
def get_ids(no_pert, pert, ids):
    no_pert = no_pert[:, 7]
    isin_pert_halo = np.isin(no_pert, ids)
    ids_perthalo = np.where(isin_pert_halo)
    common_ids = pert[:, 7][ids_perthalo]
    print("the shape of common_ids", common_ids.shape)
    print("first 10 common ids", common_ids[:10])
    return ids_perthalo, common_ids
def test_ids(common_ids, pert):
    isin_pert = np.isin(common_ids, pert[:, 7])
    # check if there is any value that is not true
    if np.any(isin_pert == False):
        print("there are some ids that are not in the perturbed halo")
        # print the ids that are not in the perturbed halo
        print(common_ids[isin_pert == False])
    else:
        print("all ids are in the perturbed halo")
def sel2(pert_sel1,m, a, b1, b2):
    line_1 = m * (pert_sel1[:, 8] - a) + b1
    line_2 = m * (pert_sel1[:, 8] - a) + b2
    print("the shape of line_1", line_1.shape)
    print("the shape of line_2", line_2.shape)
    select2_part = np.where((pert_sel1[:, 10] > line_1) & (pert_sel1[:, 10] < line_2) & (pert_sel1[:, 8] > 50))
    print("the shape of select2_part", select2_part[0].shape)
    return select2_part
def selected_ids(pert_sel1,sel2_part, no_pert, pert):
    # select the ids of the particles that are in the selected region
    #specify the size of the random slice
    slice_size = 10
    ids_sel2 = pert_sel1[:, 7][sel2_part]
    isin_pert = np.isin(pert[:, 7], ids_sel2)
    random_slice = np.random.choice(len(isin_pert), slice_size, replace=False)
    # print("random slice of array isin_pert", isin_pert[random_slice])
    ids_pert = np.where(isin_pert == True)
    # print("the shape of ids_pert", ids_pert[0].shape)
    wake_pert = pert[ids_pert]
    random_wake_part = np.random.choice(len(wake_pert), slice_size, replace=False)
    ids_nopert = np.where(np.isin(no_pert[:,7], ids_sel2) == True)
    wake_nopert = no_pert[ids_nopert]
    return wake_pert, wake_nopert

# def density_contrast(pert, no_pert):
#     # calculate the density contrast of the perturbed halo divided by the density contrast of the unperturbed halo
#     hist = np.histogram2d(pert[:, 1], pert[:, 2], bins=100)
#     hist_nopert = np.histogram2d(no_pert[:, 1], no_pert[:, 2], bins=100)
#     density_contrast = hist[0] / hist_nopert[0]
#     density = go.Heatmap(z=density_contrast.T, colorscale='Viridis')
#     fig = go.Figure(data=density)
#     fig.show()
#     fig.update_layout(
#         title="Density contrast of the perturbed halo",
#         xaxis_title="y",
#         yaxis_title="z",
#         x_axis=dict(
#             scaleanchor="y",
#             scaleratio=1),
#         y_axis=dict(
#             scaleanchor="z",
#             scaleratio=1),
#         font=dict(
#             family="Courier New, monospace",
#             size=18,
#             color="RebeccaPurple"
#     ))
#     return density_contrast

# def density_contrast(pert, no_pert):
#     # calculate the density contrast of the perturbed halo divided by the density contrast of the unperturbed halo
#     hist, xedges, yedges = np.histogram2d(pert[:, 1], pert[:, 2], bins=100)
#     hist_nopert, xedges, yedges = np.histogram2d(no_pert[:, 1], no_pert[:, 2], bins=100)
#     try:
#         density_contrast = hist / hist_nopert
#     except: # if there is a division by zero
#         density_contrast = hist / (hist_nopert + 1)
#     # plot the density contrast as a heatmap
#     fig = go.Figure(data=go.Heatmap(z=density_contrast.T, x=xedges, y=yedges, colorscale='Viridis', zmin=0, zmax=np.max(density_contrast)[0])))
#
#     # set the axis titles and plot title
#     fig.update_layout(
#         title="Density contrast of the perturbed halo",
#         xaxis_title="y",
#         yaxis_title="z",
#         font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
#     )
#
#     # add a colorbar to show the density scale
#     fig.update_layout(coloraxis=dict(colorbar=dict(title="Density")))
#
#     fig.show()
#
#     return density_contrast