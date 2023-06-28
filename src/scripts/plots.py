import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy.stats as st
from scipy.stats import skew
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter



def hist_r_l(halo,ispert=False,figname='mw'):
    """Plot the histogram of the position of the LMC around the MW.

    Parameters
    ----------
    pos : array
        Position of the LMC around the MW.

    Returns
    -------
    hist_L_pos : plot
        Histogram of the position of the LMC around the MW.
    """
    pos = halo[:, 8]
    mom = halo[:, 10]
    print(np.shape(pos))
    print(np.shape(mom))
    r = np.arange(50, 250, 10)
    L = np.arange(9, 50000, 2500)
    hist_r_L_rot = np.zeros((len(r), len(L)))
    for i in range(0, len(r) - 1):
        r_index = np.where((pos > r[i]) & (pos < r[i + 1]))
        for j in range(0, len(L) - 1):
            Nl = np.where((mom[r_index] > L[j]) & (mom[r_index] < L[j + 1]))
            hist_r_L_rot[i, j] = np.shape(Nl)[1]
    # Figure using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_ylabel(
        r"Initial angular momentum magnitude  [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]",
        fontsize=8,
    )
    ax.set_xlabel("Distance from halo center [kpc]", fontsize=10)
    if ispert == False:
        plt.suptitle(r"$\mathrm{L}_{\mathrm{mag}}(\mathrm{r}_{\mathrm{mag}})$ unperturbed halo", fontsize=15)
    elif ispert == True:
        plt.suptitle(r"$\mathrm{L}_{\mathrm{mag}}(\mathrm{r}_{\mathrm{mag}})$ perturbed halo", fontsize=15)
    im = plt.imshow(
        np.log10(hist_r_L_rot.T),
        extent=[50, 250, 0, 50000],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=5,
        cmap="Greys",
    )
    plt.tight_layout()
    plt.savefig('../../media/imgs/' + 'Lmag_pos' +figname+ '.png', bbox_inches='tight', dpi=300)
    plt.show()
    return im
    

def hist_proj_r_l(halo, proj):
    pos = halo[:, 0]
    mom = halo[:, 11]
    r = np.arange(50, 250, 10)
    L = np.arange(9, 50000, 2500)
    hist_r_L_rot = np.zeros((len(r), len(L)))
    for i in range(0, len(r) - 1):
        r_index = np.where((pos > r[i]) & (pos < r[i + 1]))
        for j in range(0, len(L) - 1):
            Nl = np.where((mom[r_index] > L[j]) & (mom[r_index] < L[j + 1]))
            hist_r_L_rot[i, j] = np.shape(Nl)[1]
    # Figure using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_ylabel(
        r"$ \mathrm{L}_x [ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}]$",
        fontsize=8,
    )
    ax.set_xlabel(
        r"$\mathrm{r}_x [ \mathrm{kpc}]$",
        fontsize=10,
    )
    
    plt.suptitle(r"$\mathrm{L}_{\mathrm{x}}(\mathrm{r}_{\mathrm{x}})$ unperturbed", fontsize=15)

    im = plt.imshow(
        np.log10(hist_r_L_rot.T),
        extent=[50, 250, 0, 50000],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=5,
        cmap="Greys",
    )
    plt.tight_layout()
    plt.show()
    return im

def calculate_histogram(halo, bins_r, bins_L):
    hist, _, _ = np.histogram2d(halo[:,8], halo[:,10], bins=[bins_r, bins_L])
    return hist

def plot_histogram(hist, figure_name):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5),dpi=600)
    im = plt.imshow(
        np.log10(hist.T),
        extent=[50, 195, 0, 50000],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=5,
        cmap="inferno",
    )
    ax.set_xlabel(r"Distance magnitudes  in [$\mathrm{kpc}$]", fontsize=12)
    ax.set_ylabel(
        r"$ \mathrm{L_{mag}}[\mathrm{low}] \,\mathrm{in} \,[ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}]$",
        fontsize=12,
    )
    plt.suptitle(r"$\mathrm{L}_{\mathrm{mag}}(\mathrm{r}_{\mathrm{mag}})[\mathrm{low}]$ unperturbed", fontsize=15)
    
    # Add dashed line at y=10000
    ax.axhline(y=10000, color='white', linestyle='--', linewidth=1)

    # Add gridlines
    ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log-scaled bin values", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig('../../media/imgs/'+'hist'+figure_name+'.png', bbox_inches='tight', dpi=300)
    plt.show()



def hist_L_r_low(pos, ang_m,figname):
    r = np.arange(50, 200, 10)
    L = np.arange(9, 50000, 2500)
    hist_r_L = np.zeros((len(r), len(L)))
    for i in range(0, len(r) - 1):
        r_index = np.where((pos > r[i]) & (pos < r[i + 1]))
        for j in range(0, len(L) - 1):
            Nl = np.where((ang_m[r_index] > L[j]) & (ang_m[r_index] < L[j + 1]))
            hist_r_L[i, j] = np.shape(Nl)[1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = plt.imshow(
        np.log10(hist_r_L.T),
        extent=[50, 195, 0, 50000],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=5,
        cmap="Greys",
    )
    ax.set_xlabel(r"Distance magnitudes  in [$\mathrm{kpc}$]", fontsize=8)
    ax.set_ylabel(
        r"$ \mathrm{L_{mag}}[\mathrm{low}] \,\mathrm{in} \,[ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}]$",
        fontsize=8,
    )
    plt.suptitle(r"$\mathrm{L}_{\mathrm{mag}}(\mathrm{r}_{\mathrm{mag}})[\mathrm{low}]$ unperturbed", fontsize=15)
    # ax.set_title("Partículas con momento angular inicial bajo", fontsize=12, y=1.05)
    plt.savefig('../../media/imgs/'+'histlow'+figname+'.png', bbox_inches='tight', dpi = 300)
    plt.show()
    return hist_L_r_low


def surface_plot(orb_pos, n_vector):
    """Plot the orbit of the LMC around the MW.

    Parameters
    ----------
    orb_pos : array
        Position of the LMC around the MW.
    n_vector : array
        Normal vector to the plane of the orbit.

    Returns
    -------
    surface_plot : plot
        Surface plot of the orbit of the LMC around the MW.
    """
    X = orb_pos[:, 0:2]
    a, b, c = n_vector
    x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10)
    y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    Z_mesh = (-a * X_mesh - b * Y_mesh) / c
    fig = go.Figure(
        data=[
            go.Surface(z=Z_mesh, x=X_mesh, y=Y_mesh, colorscale="Viridis", opacity=0.7)
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=orb_pos[:, 0],
            y=orb_pos[:, 1],
            z=orb_pos[:, 2],
            mode="markers",
            marker=dict(size=1, color="#C70039"),
        )
    )
    # fig.add_trace(
    #     go.Scatter3d(x=[0, normal_vector_line[0]], y=[0, normal_vector_line[1]], z=[0, normal_vector_line[2]],
    #                  mode='lines',
    #                  line=dict(width=5)))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10),
    )
    return fig


def LMC_plot(orb_pos, proj):
    """Plot the orbit of the LMC around the MW.

    Parameters
    ----------
    orb_pos : array
        Position of the LMC around the MW.

    Returns
    -------
    LMC_plot : plot
        Plot of the orbit of the LMC around the MW.
    """
    if proj == "xy":
        initial_point = orb_pos[0, 0:3]
        last_point = orb_pos[-1, 0:3]
        v_0 = orb_pos[0, 3:6]
        v_f = orb_pos[-1, 3:6]
        LMC_xy = go.Figure(
            data=[go.Line(x=orb_pos[:, 0], y=orb_pos[:, 1], name="LMC orbit rot xy")]
        )
        LMC_xy.add_trace(
            go.Scatter(
                x=[initial_point[0]],
                y=[initial_point[1]],
                mode="markers",
                name="initial point",
                marker=dict(size=10, color="orange"),
            )
        )
        LMC_xy.add_trace(
            go.Scatter(
                x=[last_point[0]],
                y=[last_point[1]],
                mode="markers",
                name="last point",
                marker=dict(size=10, color="purple"),
            )
        )
        LMC_xy.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            width=500,
            height=500,
            autosize=False,
            title_text="LMC orbit rot XY",
            title_x=0.5,
            title_y=0.9,
            title_font_size=15,
            title_xanchor="center",
            title_yanchor="top",
        )
        LMC_xy.update_xaxes(title_text="X [kpc]", title_font_size=15)
        LMC_xy.update_yaxes(title_text="Y [kpc]", title_font_size=15)
        LMC_xy.show()
    elif proj == "xz":
        initial_point = orb_pos[0, 0:3]
        last_point = orb_pos[-1, 0:3]
        v_0 = orb_pos[0, 3:6]
        v_f = orb_pos[-1, 3:6]
        LMC_xz = go.Figure(
            data=[go.Line(x=orb_pos[:, 0], y=orb_pos[:, 2], name="LMC orbit rot xz")]
        )
        LMC_xz.add_trace(
            go.Scatter(
                x=[initial_point[0]],
                y=[initial_point[2]],
                mode="markers",
                name="initial point",
                marker=dict(size=10, color="orange"),
            )
        )
        LMC_xz.add_trace(
            go.Scatter(
                x=[last_point[0]],
                y=[last_point[2]],
                mode="markers",
                name="last point",
                marker=dict(size=10, color="purple"),
            )
        )
        LMC_xz.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            width=500,
            height=500,
            autosize=False,
            title_text="LMC orbit rot XZ",
            title_x=0.5,
            title_y=0.9,
            title_font_size=20,
            title_xanchor="center",
            title_yanchor="top",
        )
        LMC_xz.update_xaxes(title_text="X [kpc]", title_font_size=15)
        LMC_xz.update_yaxes(title_text="Z [kpc]", title_font_size=15)
        LMC_xz.show()
    elif proj == "yz":
        initial_point = orb_pos[0, 0:3]
        last_point = orb_pos[-1, 0:3]
        velocity = orb_pos[0, 3:6]
        LMC_yz = go.Figure(
            data=[go.Line(x=orb_pos[:, 1], y=orb_pos[:, 2], name="LMC orbit rot yz")]
        )
        LMC_yz.add_trace(
            go.Scatter(
                x=[initial_point[1]],
                y=[initial_point[2]],
                mode="markers",
                name="initial point",
                marker=dict(size=10, color="orange"),
            )
        )
        LMC_yz.add_trace(
            go.Scatter(
                x=[last_point[1]],
                y=[last_point[2]],
                mode="markers",
                name="last point",
                marker=dict(size=10, color="purple"),
            )
        )
        LMC_yz.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            width=500,
            height=500,
            autosize=False,
            title_text="LMC orbit rot YZ",
            title_x=0.5,
            title_y=0.9,
            title_font_size=20,
            title_xanchor="center",
            title_yanchor="top",
        )
        LMC_yz.update_xaxes(title_text="Y [kpc]", title_font_size=15)
        LMC_yz.update_yaxes(title_text="Z [kpc]", title_font_size=15)
        LMC_yz.show()


# # Function to compute and plot density countours
def comparison_density_contour_plt(halo1, halo2, lmc, proj):
    if proj == "yz":
        title = "MW LMC perturbed halo YZ"
        figname = "density_contour_yz"
        x_label = "y [kpc]"
        y_label = "z [kpc]"
        x_data = 1
        y_data = 2
    elif proj == "xz":
        title = "MW LMC perturbed halo XZ"
        figname = "density_contour_xz"
        x_label = "x [kpc]"
        y_label = "z [kpc]"
        x_data = 0
        y_data = 2
    elif proj == "xy":
        title = "MW LMC perturbed halo XY"
        figname = "density_contour_xy"
        x_label = "x [kpc]"
        y_label = "y [kpc]"
        x_data = 0
        y_data = 1
    # Calculate min and max
    xmin_1, xmax_1 = halo1[:, x_data].min(), halo1[:, x_data].max()
    ymin_1, ymax_1 = halo1[:, y_data].min(), halo1[:, y_data].max()

    xmin_2, xmax_2 = halo2[:, x_data].min(), halo2[:, x_data].max()
    ymin_2, ymax_2 = halo2[:, y_data].min(), halo2[:, y_data].max()
    
    #Calculate min and max for both halos
    xmin = min(xmin_1, xmin_2)
    xmax = max(xmax_1, xmax_2)
    ymin = min(ymin_1, ymin_2)
    ymax = max(ymax_1, ymax_2)
    # Create a meshgrid for the contour plot
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # Estimate the PDF using Gaussian kernel density estimation
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values1 = np.vstack([halo1[:, x_data], halo1[:, y_data]])
    values2 = np.vstack([halo2[:, x_data], halo2[:, y_data]])
    kernel1 = st.gaussian_kde(values1)
    kernel2 = st.gaussian_kde(values2)
    pdf1 = np.reshape(kernel1(positions).T, xx.shape)
    pdf2 = np.reshape(kernel2(positions).T, xx.shape)

    # Calculate global minimum and maximum values for both PDFs
    global_min = min(pdf1.min(), pdf2.min())
    global_max = max(pdf1.max(), pdf2.max())

    # # Normalize the color map using the global minimum and maximum values
    # norm = Normalize(vmin=global_min, vmax=global_max)

    # # Normalize the color map using the LogNorm function
    # norm = LogNorm(vmin=global_min, vmax=global_max)
    # Normalize the color map using the SymLogNorm function
    # You can adjust the `linthresh` parameter to control the range around zero that is mapped linearly
    norm = SymLogNorm(linthresh=0.01, vmin=global_min, vmax=global_max)

    # Define the number of levels for the contour plots
    num_levels = 10
    levels = np.linspace(global_min, global_max, num_levels)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=600, sharex=True, sharey=True)
    

    # Contour plot for the first subplot
    contour_filled1 = ax1.contourf(xx, yy, pdf1, cmap="viridis", norm=norm, levels=levels)
    # ax1.contour(xx, yy, pdf1, colors="black", alpha=0.8, linewidths=0.5)
    #LMC orbit plot in the projection
    ax1.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="#E74C3C",
        alpha=0.8,
        label="lmc",
        marker="o",
        markersize=3,
        markerfacecolor='#E7AC3C',
        markeredgecolor='#E78C3C'
    )
    # ax1.set_title('Unperturbed halo '+proj)  # Add title for the first subplot
    # ax1.set_xlabel(x_label)  # Add x-axis label for the first subplot
    # ax1.set_ylabel(y_label)  # Add y-axis label for the first subplot
    # Set zoom
    ax1.set_xlim(xmin_2, xmax_2)
    ax1.set_ylim(ymin_2, ymax_2)
    # Set equal aspect ratio
    ax1.set_aspect("equal", "box")
    ax1.set_title('Unperturbed halo '+proj, fontsize=16)
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True)

    
    
    
    contour_filled2 = ax2.contourf(xx, yy, pdf2, cmap="viridis", norm=norm, levels=levels)
    ax2.contour(xx, yy, pdf2, colors="black", alpha=0.8, linewidths=1.5, levels=levels)
        #LMC orbit plot in the projection
    ax2.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="#E74C3C",
        alpha=0.8,
        label="lmc",
        marker="o",
        markersize=3,
        markerfacecolor='#E7AC3C',
        markeredgecolor='#E78C3C'
    )
    
    ax2.set_title('Perturbed halo '+proj, fontsize=16)
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(y_label, fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_xlim(xmin_2, xmax_2)
    ax2.set_ylim(ymin_2, ymax_2)
    ax2.set_aspect("equal", "box")
    ax2.grid(True)

    # ax2.set_title('Perturbed halo '+proj)  # Add title for the second subplot
    # ax2.set_xlabel(x_label)  # Add x-axis label for the second subplot
    # ax2.set_ylabel(y_label)  # Add y-axis label for the second subplot
    # Set zoom
    # ax2.set_xlim(xmin_2, xmax_2)
    # ax2.set_ylim(ymin_2, ymax_2)
    # ax2.set_aspect("equal", "box")
    # Create a single colorbar for both subplots
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
        # Add a colorbar to the plot
    cbar = fig.colorbar(contour_filled2, ax=[ax1, ax2], format=ScalarFormatter())
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Density', fontsize=14)

    
    
    # fig.colorbar(contour_filled2, cax=cax)
    # Add a general title for the entire figure
    fig.suptitle('Density Contours in proj '+proj, fontsize=24)
    plt.savefig('../../media/imgs/sel2/' + 'Lmag_pos' +figname+ '.png', bbox_inches='tight', dpi=300)

    plt.show()
    #     # Set the figure size
    #     fig, ax = plt.subplots(figsize=(8, 8))


#     # Set the aspect ratio to 'equal'
#     ax.set_aspect("equal")
#     # Contour plot
#     # Contour filled
#     contour_filled = ax.contourf(xx, yy, pdf, cmap="viridis")
#     # Contour lines
#     ax.contour(xx, yy, pdf, colors="black", alpha=0.8, linewidths=0.5)
#     # Add a colorbar
#     fig.colorbar(contour_filled, ax=ax, shrink=0.8, extend="both")

#     # # Scatter plot
#     # ax.scatter(halo[:, x_data], halo[:, y_data], s=5, c='white', alpha=0.5)

#     # Set labels and title
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("Particle Density Contour Plot (x-y projection)")

#     # Display the plot
#     plt.show()

# Function to compute and plot density countours
def density_contour_plt(halo, lmc, proj):
    if proj == "yz":
        title = "MW LMC perturbed halo YZ"
        figname = "density_contour_yz"
        x_label = "y [kpc]"
        y_label = "z [kpc]"
        x_data = 1
        y_data = 2
    elif proj == "xz":
        title = "MW LMC perturbed halo XZ"
        figname = "density_contour_xz"
        x_label = "x [kpc]"
        y_label = "z [kpc]"
        x_data = 0
        y_data = 2
    elif proj == "xy":
        title = "MW LMC perturbed halo XY"
        figname = "density_contour_xy"
        x_label = "x [kpc]"
        y_label = "y [kpc]"
        x_data = 0
        y_data = 1
    # Calculate min and max
    xmin, xmax = halo[:, x_data].min(), halo[:, x_data].max()
    ymin, ymax = halo[:, y_data].min(), halo[:, y_data].max()

    # Create a meshgrid for the contour plot
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # Estimate the PDF using Gaussian kernel density estimation
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([halo[:, x_data], halo[:, y_data]])
    kernel = st.gaussian_kde(values)
    pdf = np.reshape(kernel(positions).T, xx.shape)

    # Set the figure size
    fig, ax = plt.subplots(figsize=(8, 8))

    # LMC orbit plot in the projection
    ax.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="cyan",
        alpha=0.8,
        label="lmc",
    )
    # Set the aspect ratio to 'equal'
    ax.set_aspect("equal")
    # Contour plot
    # Contour filled
    contour_filled = ax.contourf(xx, yy, pdf, cmap="viridis")
    # Contour lines
    ax.contour(xx, yy, pdf, colors="black", alpha=0.8, linewidths=0.5)
    # Add a colorbar
    fig.colorbar(contour_filled, ax=ax, shrink=0.8, extend="both")

    # # Scatter plot
    # ax.scatter(halo[:, x_data], halo[:, y_data], s=5, c='white', alpha=0.5)

    # Set labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Particle Density Contour Plot (x-y projection)")

    # Display the plot
    plt.show()

def comparison_hist_orbit_plt(halo1, halo2,lmc, proj, coarse_step, arrow_scale, arrow_width):
    """
    Create a comparison plot of two halos with quiver plots and 2D histograms.

    Parameters:
    - halo1: First halo dataset
    - halo2: Second halo dataset
    - lmc: Large Magellanic Cloud dataset
    - proj: Projection type ('yz', 'xz', or 'xy')
    - coarse_step: Step size for coarser grid in quiver plot
    - arrow_scale: Scale for arrows in quiver plot
    - arrow_width: Width for arrows in quiver plot
    """
    # Define projection-specific settings
    if proj == "yz":
        title = "MW LMC perturbed halo YZ"
        figname = "hist_orb_pert_yz"
        x_label = "y [kpc]"
        y_label = "z [kpc]"
        x_data = 1
        y_data = 2
        vx_data = 4
        vy_data = 5
    elif proj == "xz":
        title = "MW LMC perturbed halo XZ"
        figname = "hist_orb_pert_xz"
        x_label = "x [kpc]"
        y_label = "z [kpc]"
        x_data = 0
        y_data = 2
        vx_data = 3
        vy_data = 5
    elif proj == "xy":
        title = "MW LMC perturbed halo XY"
        figname = "hist_orb_pert_xy"
        x_label = "x [kpc]"
        y_label = "y [kpc]"
        x_data = 0
        y_data = 1
        vx_data = 3
        vy_data = 4
    else:
        raise ValueError("Invalid projection")
    # Create a coarser grid for the starting positions
    # Adjust this value to control the density of arrows
    x1_positions = halo1[::coarse_step, x_data]
    y1_positions = halo1[::coarse_step, y_data]
    x1_directions = halo1[::coarse_step, vx_data]
    y1_directions = halo1[::coarse_step, vy_data]

    x2_positions = halo2[::coarse_step, x_data]
    y2_positions = halo2[::coarse_step, y_data]
    x2_directions = halo2[::coarse_step, vx_data]
    y2_directions = halo2[::coarse_step, vy_data]

    #Get the min and max values of the halo 
    # Calculate the 2D histograms and normalize them
    hist1, xedges, yedges = np.histogram2d(halo1[:, x_data], halo1[:, y_data], bins=100)
    hist2, _, _ = np.histogram2d(halo2[:, x_data], halo2[:, y_data], bins=(xedges, yedges))
    hist1 = hist1 / np.max(hist1)
    hist2 = hist2 / np.max(hist2)

    # Set up the colormap and calculate global min and max values
    cmap = plt.get_cmap("viridis")
    global_min = min(hist1.min(), hist2.min())
    global_max = max(hist1.max(), hist2.max())

    # Calculate the min and max values of halo 2 for x and y axes
    x_min_halo2 = np.min(halo2[:, x_data])
    x_max_halo2 = np.max(halo2[:, x_data])
    y_min_halo2 = np.min(halo2[:, y_data])
    y_max_halo2 = np.max(halo2[:, y_data])


    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # First subplot: quiver plot and 2D histogram
    ax1.quiver(x1_positions, y1_positions, x1_directions, y1_directions, scale=arrow_scale, width=arrow_width, color="white", alpha=0.5, label='label', edgecolors="black", linewidths=0.5)
    im1 = ax1.imshow(hist1.T, origin="lower", cmap=cmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", vmin=global_min, vmax=global_max)
    ax1.set_title('Unperturbed halo')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    # Set the x and y limits for the first subplot based on halo 2 min and max values
    ax1.set_xlim(x_min_halo2, x_max_halo2)
    ax1.set_ylim(y_min_halo2, y_max_halo2)
    # Set the aspect ratio to be equal for the first subplot while keeping xlim and ylim values
    ax1.set_aspect('equal', adjustable='box')   
    # LMC orbit plot in the projection
    ax1.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="#E74C3C",
        alpha=0.8,
        label="lmc",
        marker="o",
        markersize=3,
        markerfacecolor='#E7AC3C',
        markeredgecolor='#E78C3C'
    )
    # Customize grid
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # Calculate the skewness of the halo
    unpert_skew_x = skew(hist1, axis=0)
    unpert_skew_y = skew(hist1, axis=1)
    
    # Calculate the angles of the arrows in radians
    angles1 = np.arctan2(y1_directions, x1_directions)
    # Calculate the mean and stardard deviation of the angles
    mean_angles1 = np.mean(angles1)
    std_angles1 = np.std(angles1)
    print("Mean angle: ", mean_angles1)
    print("Standard deviation of angles: ", std_angles1)
    # Calculate the mean direction vector
    mean_cos1 = np.mean(np.cos(angles1))
    mean_sin1 = np.mean(np.sin(angles1))
    mean_direction_vector = np.array([mean_cos1, mean_sin1])
    
    #Calculate the length of the mean direction vector
    mean_direction_vector_length = np.linalg.norm(mean_direction_vector)
    
    #Calculate the circular variance
    circular_variance = 1 - mean_direction_vector_length
    print("Circular variance: ", circular_variance)

    # Calculate the gradient of the arrows with respect to the x and y coordinates
    grad_x1= np.gradient(x1_directions)
    grad_y1= np.gradient(y1_directions)
    
    # Calculate the magnitude of the gradient vectors for the first plot
    magnitude_grad_x1 = np.sqrt(grad_x1[0]**2 + grad_x1[1]**2)
    magnitude_grad_y1 = np.sqrt(grad_y1[0]**2 + grad_y1[1]**2)

    # Calculate the average magnitude of the gradient vectors for the first plot
    avg_magnitude_grad_x1 = np.mean(magnitude_grad_x1)
    avg_magnitude_grad_y1 = np.mean(magnitude_grad_y1)

    # Calculate the divergence of the velocity field for the first plot
    divergence1 = grad_x1[0] + grad_y1[1]
    print("Divergence: ", divergence1)
    

    # Second subplot: quiver plot and 2D histogram
    ax2.quiver(x2_positions, y2_positions, x2_directions, y2_directions, scale=arrow_scale, width=arrow_width, color="white", alpha=0.5, label='label2', edgecolors="black", linewidths=0.5)
    im2 = ax2.imshow(hist2.T, origin="lower", cmap=cmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", vmin=global_min, vmax=global_max)
    ax2.set_title('Perturbed halo')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    # Set the x and y limits for the first subplot based on halo 2 min and max values
    ax2.set_xlim(x_min_halo2, x_max_halo2)
    ax2.set_ylim(y_min_halo2, y_max_halo2)
    # LMC orbit plot in the projection
    # Set the aspect ratio to be equal for the second subplot while keeping xlim and ylim values    
    ax2.set_aspect('equal', adjustable='box')
    ax2.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="#E74C3C",
        alpha=0.8,
        label="lmc",
        marker="o",
        markersize=3,
        markerfacecolor='#E7AC3C',
        markeredgecolor='#E78C3C'
    )
    
    # Customize grid
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # Calculate the skewness of the halo
    pert_skew_x = skew(hist2, axis=0)
    pert_skew_y = skew(hist2, axis=1)

    # Add a colorbar for both subplots
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax)

    unpert_mean_skew_x = np.mean(unpert_skew_x)
    unpert_mean_skew_y = np.mean(unpert_skew_y)
    pert_mean_skew_x = np.mean(pert_skew_x)
    pert_mean_skew_y = np.mean(pert_skew_y)
    
    print(f"Skewness of unperturbed halo in{proj[0]} direction: {unpert_mean_skew_x}")
    print(f"Skewness of unperturbed halo in {proj[1]} direction: {unpert_mean_skew_y}")
    print(f"Skewness of perturbed halo in {proj[0]} direction: {pert_mean_skew_x}")
    print(f"Skewness of perturbed halo in {proj[1]} direction: {pert_mean_skew_y}")
    
    
    # Calculate the angles of the arrows in radians
    angles2 = np.arctan2(y2_directions, x2_directions)
    # Calculate the mean and stardard deviation of the angles
    mean_angles2 = np.mean(angles2)
    std_angles2 = np.std(angles2)
    print("Mean angle: ", mean_angles2)
    print("Standard deviation of angles: ", std_angles2)
    # Calculate the mean direction vector
    mean_cos2 = np.mean(np.cos(angles2))
    mean_sin2 = np.mean(np.sin(angles2))
    mean_direction_vector = np.array([mean_cos2, mean_sin2])
    
    #Calculate the length of the mean direction vector
    mean_direction_vector_length = np.linalg.norm(mean_direction_vector)
    
    #Calculate the circular variance
    circular_variance = 1 - mean_direction_vector_length
    print("Circular variance: ", circular_variance)    
    
        # Calculate the gradient of the arrows with respect to the x and y coordinates
    grad_x2= np.gradient(x2_directions)
    grad_y2= np.gradient(y2_directions)
    # Calculate the magnitude of the gradient vectors for the second plot
    magnitude_grad_x2 = np.sqrt(grad_x2[0]**2 + grad_x2[1]**2)
    magnitude_grad_y2 = np.sqrt(grad_y2[0]**2 + grad_y2[1]**2)

    # Calculate the average magnitude of the gradient vectors for the second plot
    avg_magnitude_grad_x2 = np.mean(magnitude_grad_x2)
    avg_magnitude_grad_y2 = np.mean(magnitude_grad_y2)

    # Calculate the divergence of the velocity field for the second plot
    divergence2 = grad_x2 + grad_y2
    print("Divergence: ", divergence2)
    print("gradient x shape", grad_x2.shape)
    print("gradient y shape", grad_y2.shape)
    # Compare the average magnitudes of the gradient vectors
    if avg_magnitude_grad_x1 > avg_magnitude_grad_x2 and avg_magnitude_grad_y1 > avg_magnitude_grad_y2:
        print("The first plot has a higher gradient.")
    elif avg_magnitude_grad_x1 < avg_magnitude_grad_x2 and avg_magnitude_grad_y1 < avg_magnitude_grad_y2:
        print("The second plot has a higher gradient.")
    else:
        print("The gradients are not clearly higher in one plot.")
    
    # Add a general title for the entire figure
    fig.suptitle('Particle density and velocity direction for proj'+proj, fontsize=16)
    
    # Save the figure
    plt.savefig('../../media/imgs/sel2/' +figname+ '.png', bbox_inches='tight', dpi=300)
    
    plt.show()
    

# Function to plot histogram of sel2 particles plus LMC orbit
def hist_orbit_plt(halo, lmc, proj, coarse_step, arrow_scale, arrow_width, ispert):
    if ispert == True:
        label = "pert"
        if proj == "yz":
            title = "MW LMC perturbed halo YZ"
            figname = "hist_orb_pert_yz"
            x_label = "y [kpc]"
            y_label = "z [kpc]"
            x_data = 1
            y_data = 2
            vx_data = 4
            vy_data = 5
        elif proj == "xz":
            title = "MW LMC perturbed halo XZ"
            figname = "hist_orb_pert_xz"
            x_label = "x [kpc]"
            y_label = "z [kpc]"
            x_data = 0
            y_data = 2
            vx_data = 3
            vy_data = 5
        elif proj == "xy":
            title = "MW LMC perturbed halo XY"
            figname = "hist_orb_pert_xy"
            x_label = "x [kpc]"
            y_label = "y [kpc]"
            x_data = 0
            y_data = 1
            vx_data = 3
            vy_data = 4
        else:
            raise ValueError("Invalid projection")
    elif ispert == False:
        label = "unpert"
        if proj == "yz":
            title = "MW LMC unperturbed halo YZ"
            figname = "hist_orb_unpert_yz"
            x_label = "y [kpc]"
            y_label = "z [kpc]"
            x_data = 1
            y_data = 2
            vx_data = 4
            vy_data = 5
        elif proj == "xz":
            title = "MW LMC unperturbed halo XZ"
            figname = "hist_orb_unpert_xz"
            x_label = "x [kpc]"
            y_label = "z [kpc]"
            x_data = 0
            y_data = 2
            vx_data = 3
            vy_data = 5
        elif proj == "xy":
            title = "MW LMC unperturbed halo XY"
            figname = "hist_orb_unpert_xy"
            x_label = "x [kpc]"
            y_label = "y [kpc]"
            x_data = 0
            y_data = 1
            vx_data = 3
            vy_data = 4
        else:
            raise ValueError("Invalid projection")
    # Create a coarser grid for the starting positions
    # Adjust this value to control the density of arrows
    x_positions = halo[::coarse_step, x_data]
    y_positions = halo[::coarse_step, y_data]
    x_directions = halo[::coarse_step, vx_data]
    y_directions = halo[::coarse_step, vy_data]

    fig, ax = plt.subplots()
    ax.quiver(
        x_positions,
        y_positions,
        x_directions,
        y_directions,
        scale=arrow_scale,
        width=arrow_width,
        color="white",
        alpha=0.5,
        label=label,
        edgecolors="black",
        linewidths=0.5,
    )

    # ax.scatter(pert[:, x_data], pert[:, y_data], s=1, color='red', alpha=0.8, label='pert')

    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(halo[:, x_data], halo[:, y_data], bins=30)

    # Normalize the histogram
    hist = hist / np.max(hist)

    # Create a colormap
    cmap = plt.get_cmap("viridis")

    # Plot the histogram using imshow
    im = ax.imshow(
        hist.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, label="Particle density", shrink=0.5)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Get the x-axis and y-axis limits from the histogram plot
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # LMC orbit plot in the projection
    ax.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="cyan",
        alpha=0.8,
        label="lmc",
    )
    # ax.scatter(pert[:, x_data], pert[:, y_data], s=2, color='violet', alpha=0.8, label='pert pos')
    # Get the x-axis and y-axis limits from the line plot

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    arrow_start = (lmc[:, x_data][-1], lmc[:, y_data][-1])
    arrow_end = (lmc[:, x_data][-1] + 0.5, lmc[:, y_data][-1])
    ax.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops=dict(arrowstyle="->", lw=3, color="cyan", alpha=0.8),
    )

    ax.set_title(title)

    ax.legend()
    ax.set_aspect("equal")

    # save the plot to a high resolution image
    plt.savefig("../../media/imgs/sel2/" + figname + ".png", dpi=300)

    # Show the plot
    plt.show()


# Function to plot vector field of sel2 pos particles
def vector_field_plt(halo, lmc, proj, coarse_step, arrow_scale, arrow_width, ispert):
    if ispert == True:
        label = "pert"
        if proj == "yz":
            title = "velocity field perturbed halo sel 2 YZ"
            x_label = "y [kpc]"
            y_label = "z [kpc]"
            x_data = 1
            y_data = 2
            vx_data = 4
            vy_data = 5
            fig_name = "vel_pert_sel2_yz"
        elif proj == "xz":
            title = "velocity field perturbed halo sel 2 XZ"
            x_label = "x [kpc]"
            y_label = "z [kpc]"
            x_data = 0
            y_data = 2
            vx_data = 3
            vy_data = 5
            fig_name = "vel_pert_sel2_xz"
        elif proj == "xy":
            title = "velocity field perturbed halo sel 2 XY"
            x_label = "x [kpc]"
            y_label = "y [kpc]"
            x_data = 0
            y_data = 1
            vx_data = 3
            vy_data = 4
            fig_name = "vel_pert_sel2_xy"

        else:
            raise ValueError("Invalid projection")
    elif ispert == False:
        label = "unpert"
        if proj == "yz":
            title = "velocity field unperturbed halo sel 2 YZ"
            x_label = "y [kpc]"
            y_label = "z [kpc]"
            x_data = 1
            y_data = 2
            vx_data = 4
            vy_data = 5
            fig_name = "vel_nopert_sel2_yz"
        elif proj == "xz":
            title = "velocity field unperturbed halo sel 2 XZ"
            x_label = "x [kpc]"
            y_label = "z [kpc]"
            x_data = 0
            y_data = 2
            vx_data = 3
            vy_data = 5
            fig_name = "vel_nopert_sel2_xz"
        elif proj == "xy":
            title = "velocity field unperturbed halo sel 2 XY"
            x_label = "x [kpc]"
            y_label = "y [kpc]"
            x_data = 0
            y_data = 1
            vx_data = 3
            vy_data = 4
            fig_name = "vel_nopert_sel2_xy"

        else:
            raise ValueError("Invalid projection")

        # Create a coarser grid for the starting positions

    x_positions = halo[::coarse_step, x_data]
    y_positions = halo[::coarse_step, y_data]
    x_directions = halo[::coarse_step, vx_data]
    y_directions = halo[::coarse_step, vy_data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(
        x_positions,
        y_positions,
        x_directions,
        y_directions,
        scale=arrow_scale,
        width=arrow_width,
        color="blue",
        alpha=0.5,
        label=label,
        edgecolors="black",
        linewidths=0.5,
    )

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.quiver(no_pert[:, x_data], no_pert[:, y_data],
    #           no_pert[:, vx_data], no_pert[:, vy_data],
    #           scale=1500, width=0.005, color='blue', alpha=0.8, label='no pert')

    # ax.scatter(pert[:, x_data], pert[:, y_data], s=1, color='red', alpha=0.8, label='pert')
    #

    # Set axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # LMC orbit plot in the projection
    ax.plot(
        lmc[:, x_data],
        lmc[:, y_data],
        linestyle="-",
        color="green",
        alpha=0.8,
        label="lmc",
    )
    # ax.scatter(pert[:, x_data], pert[:, y_data], s=2, color='violet', alpha=0.8, label='pert pos')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.set_aspect("equal")
    # save the plot to a high resolution image
    plt.savefig("../../media/imgs/sel2/" + fig_name + ".png", dpi=300)
    # Show the plot
    plt.show()


def hist(halo, orb, proj):
    """Plot the histogram of the distances between the LMC and the MW.

    Parameters
    ----------
    halo : array
        Position of the DM particles of MW.
    orb : array
        Orbit of the LMC around the MW.
    idx : array
        Index list of particles
    proj : str
        Projection of the orbit.
    low_L : array
        particles selected in low momenta. region

    Returns
    -------
    hist : plot
        Histogram of the distances between the LMC and the MW.
    """
    fig = go.Figure()
    if proj == "xy":
        # Create scatter plot
        fig.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 1],
                mode="markers",
                marker=dict(color="deepskyblue", size=1),
                name="MW DM Halo particles",
            )
        )

        # Create line plot
        fig.add_trace(
            go.Scatter(
                x=orb[:, 0],
                y=orb[:, 1],
                mode="lines",
                line=dict(color="chartreuse", width=1),
                name="LMC Orbit",
            )
        )

        # Set axis labels and title
        fig.update_layout(
            xaxis_title=r"x [$\mathrm{kpc}$]",
            yaxis_title=r"y [$\mathrm{kpc}$]",
            title="Proyeccion xy",
            title_font_size=14,
        )

        # Set axis tick font size
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))

        # Set the aspect ratio to 'equal'
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )
        # fig.write_image("proj_xy.png")
    elif proj == "xz":
        fig.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 2],
                mode="markers",
                marker=dict(color="deepskyblue", size=1),
                name="MW DM Halo particles",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=orb[:, 0],
                y=orb[:, 2],
                mode="lines",
                line=dict(color="chartreuse", width=1),
                name="LMC Orbit",
            )
        )
        fig.update_layout(
            xaxis_title=r"x [$\mathrm{kpc}$]",
            yaxis_title=r"z [$\mathrm{kpc}$]",
            title="Proyeccion xz",
            title_font_size=14,
        )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )
        print("paso por aqui")
        # fig.write_image("proj_xz.png")
    elif proj == "yz":
        fig.add_trace(
            go.Scatter(
                x=halo[:, 1],
                y=halo[:, 2],
                mode="markers",
                marker=dict(color="deepskyblue", size=1),
                name="MW DM Halo particles",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=orb[:, 1],
                y=orb[:, 2],
                mode="lines",
                line=dict(color="chartreuse", width=1),
                name="LMC Orbit",
            )
        )
        fig.update_layout(
            xaxis_title=r"y [$\mathrm{kpc}$]",
            yaxis_title=r"z [$\mathrm{kpc}$]",
            title="Proyeccion yz",
            title_font_size=14,
        )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )
        # fig.write_image("proj_yz.png")
    fig.show()


# def hist_pos_deltaL(pos, pos_oy, pos_oz, ids):
#     fig = go.Figure()
#     #
#     fig.add_trace(go.Scatter(x=pos[ids, 1], y=pos[ids, 2],
#                              mode='markers', marker=dict(size=5, color='deepskyblue', opacity=0.5)))
#
#     fig.add_trace(go.Scatter(x=pos_oy, y=pos_oz, mode='lines',
#                              line=dict(color='chartreuse', width=1.5)))
#
#     fig.update_xaxes(title_text='y [kpc]', title_font=dict(size=20))
#     fig.update_yaxes(title_text='z [kpc]', title_font=dict(size=20))
#
#     fig.update_layout(xaxis=dict(range=[-150, 150]), yaxis=dict(range=[-150, 150]))
#     fig.show()
#     return fig


def hist_delta_xy(pos, pos_oy, pos_oz, index_percentage):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.scatter(
        pos[index_percentage, 0],
        pos[index_percentage, 1],
        s=5,
        alpha=0.5,
        c=["deepskyblue"],
        marker=".",
    )
    plt.plot(pos_oy, pos_oz, alpha=0.8, c="chartreuse", linewidth=1.5)
    ax.set_xlabel(r"x [$\mathrm{kpc}$]", fontsize=20)
    ax.set_ylabel(r"y [$\mathrm{kpc}$]", fontsize=20)
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    # plt.title(r"Projection $xy$ high $L_$ ", fontsize=12)
    # ax.grid(which='both')
    return hist_delta_xy


def plot_3d(arr1, arr2, title):
    plot_3d = go.Figure()
    plot_3d.add_trace(
        go.Scatter3d(
            x=arr1[:, 0],
            y=arr1[:, 1],
            z=arr1[:, 2],
            mode="markers",
            marker=dict(size=1, color=arr1[:, 10], colorscale="viridis", opacity=0.8),
            name="wake",
        )
    )
    plot_3d.add_trace(
        go.Scatter3d(
            x=arr2[:, 0],
            y=arr2[:, 1],
            z=arr2[:, 2],
            mode="lines",
            marker=dict(size=1, color="orange", opacity=0.8),
            name="lmc",
        )
    )
    plot_3d.update_layout(
        scene=dict(
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
            zaxis_title="z [kpc]",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        title=title,
        title_font_size=14,
    )
    plot_3d.add_trace(
        go.Cone(
            x=[arr2[-1, 0]],
            y=[arr2[-1, 1]],
            z=[arr2[-1, 2]],
            u=[arr2[-1, 3] - arr2[-2, 3]],
            v=[arr2[-1, 4] - arr2[-2, 4]],
            w=[arr2[-1, 5] - arr2[-2, -5]],
            sizemode="absolute",
            sizeref=10,
            anchor="tail",
            showscale=False,
            opacity=0.8,
            name="lmc velocity",
        )
    )
    plot_3d.show()


def plot_3d_plt(arr1, lmc, title):
    fig = plt.figure(dpi=300, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter plot for arr1
    scatter = ax.scatter(arr1[:, 0], arr1[:, 1], arr1[:, 2], c=arr1[:, 10], cmap='viridis', s=2, alpha=0.8, label='wake', edgecolors='none')

    # 3D line plot for arr2
    ax.plot(lmc[:, 0], lmc[:, 1], lmc[:, 2], color='#C70039', alpha=0.8, linewidth=1)

    # Add a colorbar for the scatter plot
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax, shrink=0.3)
    cbar.set_label(f'$L$ [kpc km/s]', fontsize=8)
    
    #create quiver velocity plot of LMC
    ax.quiver(lmc[:, 0], lmc[:, 1], lmc[:, 2], lmc[:, 3], lmc[:, 4], lmc[:, 5], length=1, normalize=True, color='#C70039', alpha=0.8, linewidth=1, label='LMC velocity')
    
    # Set axis labels and title
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    ax.set_title(title, fontsize=12, y=1.05)
    # Adjust subplot margins
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)


    
    # Display the plot
    plt.show()

def proj_both(arr1, arr2, arr3, proj):
    if proj == "xy":
        proj_xy_both = go.Figure()
        proj_xy_both.add_trace(
            go.Scatter(
                x=arr1[:, 0],
                y=arr1[:, 1],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr1[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="halo pert",
            )
        )
        proj_xy_both.add_trace(
            go.Scatter(
                x=arr2[:, 0],
                y=arr2[:, 1],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr2[:, 10],
                    colorscale="agsunset",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="halo_no_pert",
            )
        )
        proj_xy_both.add_trace(
            go.Scatter(
                x=arr3[:, 0],
                y=arr3[:, 1],
                mode="lines",
                marker=dict(size=1, color="blue", opacity=0.8),
                name="lmc xy",
            )
        )
        proj_xy_both.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XY",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
        )
        proj_xy_both.show()
    elif proj == "xz":
        proj_xz_both = go.Figure()
        proj_xz_both.add_trace(
            go.Scatter(
                x=arr1[:, 0],
                y=arr1[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr1[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="halo pert",
            )
        )
        proj_xz_both.add_trace(
            go.Scatter(
                x=arr2[:, 0],
                y=arr2[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr2[:, 10],
                    colorscale="agsunset",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="halo_no_pert",
            )
        )
        proj_xz_both.add_trace(
            go.Scatter(
                x=arr3[:, 0],
                y=arr3[:, 2],
                mode="lines",
                marker=dict(size=1, color="blue", opacity=0.8),
                name="lmc xz",
            )
        )
        proj_xz_both.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XZ",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        proj_xz_both.show()
    elif proj == "yz":
        proj_yz_both = go.Figure()
        proj_yz_both.add_trace(
            go.Scatter(
                x=arr1[:, 1],
                y=arr1[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr1[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=5,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="halo pert",
            )
        )
        proj_yz_both.add_trace(
            go.Scatter(
                x=arr2[:, 1],
                y=arr2[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=arr2[:, 10],
                    colorscale="agsunset",
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=5,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.5,
                        xanchor="left",
                    ),
                ),
                name="halo_no_pert",
            )
        )
        proj_yz_both.add_trace(
            go.Scatter(
                x=arr3[:, 1],
                y=arr3[:, 2],
                mode="lines",
                marker=dict(size=1, color="blue", opacity=0.8),
                name="lmc yz",
            )
        )
        proj_yz_both.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel YZ",
            xaxis_title="y [kpc]",
            yaxis_title="z [kpc]",
        )
        proj_yz_both.show()


def proj_xy(arr1, arr2):
    proj_xy = go.Figure()
    proj_xy.add_trace(
        go.Scatter(
            x=arr1[:, 0],
            y=arr1[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=arr1[:, 10],
                colorscale="viridis",
                opacity=0.8,
                colorbar=dict(
                    title="L [kpc km/s]",
                    titleside="top",
                    tickmode="array",
                    tickvals=[10000, 20000, 30000],
                    ticktext=["10000", "20000", "30000"],
                    ticks="outside",
                    thickness=20,
                    len=0.5,
                    y=0.5,
                    yanchor="middle",
                    x=1.1,
                    xanchor="left",
                ),
            ),
            name="halo pert",
        )
    )
    proj_xy.add_trace(
        go.Scatter(
            x=arr2[:, 0],
            y=arr2[:, 1],
            mode="lines",
            marker=dict(size=1, color="red", opacity=0.8),
            name="lmc xy",
        )
    )
    proj_xy.update_layout(
        autosize=False,
        width=600,
        height=600,
        xaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        yaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        ),
        title="DM wake sel XY",
        xaxis_title="x [kpc]",
        yaxis_title="y [kpc]",
    )
    proj_xy.show()


def proj_yz(arr1, arr2):
    proj_yz = go.Figure()
    proj_yz.add_trace(
        go.Scatter(
            x=arr1[:, 1],
            y=arr1[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=arr1[:, 10],
                colorscale="viridis",
                opacity=0.8,
                colorbar=dict(
                    title="L [kpc km/s]",
                    titleside="top",
                    tickmode="array",
                    tickvals=[10000, 20000, 30000],
                    ticktext=["10000", "20000", "30000"],
                    ticks="outside",
                    thickness=20,
                    len=0.5,
                    y=0.5,
                    yanchor="middle",
                    x=1.1,
                    xanchor="left",
                ),
            ),
            name="halo pert",
        )
    )
    proj_yz.add_trace(
        go.Scatter(
            x=arr2[:, 1],
            y=arr2[:, 2],
            mode="lines",
            marker=dict(size=1, color="red", opacity=0.8),
            name="lmc xy",
        )
    )
    proj_yz.update_layout(
        autosize=False,
        width=600,
        height=600,
        xaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        yaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        ),
        title="DM wake sel YZ",
        xaxis_title="y [kpc]",
        yaxis_title="z [kpc]",
    )
    proj_yz.show()


def proj_xz(arr1, arr2):
    proj_xz = go.Figure()
    proj_xz.add_trace(
        go.Scatter(
            x=arr1[:, 0],
            y=arr1[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=arr1[:, 10],
                colorscale="viridis",
                opacity=0.8,
                colorbar=dict(
                    title="L [kpc km/s]",
                    titleside="top",
                    tickmode="array",
                    tickvals=[10000, 20000, 30000],
                    ticktext=["10000", "20000", "30000"],
                    ticks="outside",
                    thickness=20,
                    len=0.5,
                    y=0.5,
                    yanchor="middle",
                    x=1.1,
                    xanchor="left",
                ),
            ),
            name="halo pert",
        )
    )

    proj_xz.add_trace(
        go.Scatter(
            x=arr2[:, 0],
            y=arr2[:, 2],
            mode="lines",
            marker=dict(size=1, color="red", opacity=0.8),
            name="lmc xy",
        )
    )
    proj_xz.update_layout(
        autosize=False,
        width=600,
        height=600,
        xaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        yaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        ),
        title="DM wake sel XZ",
        xaxis_title="x [kpc]",
        yaxis_title="z [kpc]",
    )
    proj_xz.show()


def vector_field(halo, lmc, proj, slice_size):
    """
    :param halo:
    :param lmc:
    :param proj:
    :param slice_size:
    :return: vector field of the halo and lmc in the given projection
    """
    if slice_size > halo.shape[0]:
        print(
            "slice size is larger than the halo size, setting slice size to halo size"
        )
        slice_size = halo.shape[0]
    random_slice = np.random.choice(halo.shape[0], slice_size, replace=False)
    halo1 = halo[random_slice]
    if proj == "yz":
        y_i = lmc[0, 1]
        z_i = lmc[0, 2]
        vel_yz = ff.create_quiver(
            halo1[:, 1],
            halo1[:, 2],
            halo1[:, 4],
            halo1[:, 5],
            arrow_scale=0.25,
            name="vel",
            marker=dict(color="orange", size=5),
            line=dict(width=1),
        )
        vel_yz.add_trace(
            go.Scatter(
                x=lmc[:, 1],
                y=lmc[:, 2],
                mode="lines",
                marker=dict(size=2, color="blue", opacity=0.8),
                name="lmc",
            )
        )
        vel_yz.add_trace(
            go.Scatter(
                x=[y_i],
                y=[z_i],
                mode="markers",
                marker=dict(size=10, color="green", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_yz.add_trace(
            go.Scatter(
                x=halo[:, 1],
                y=halo[:, 2],
                mode="markers",
                marker=dict(size=2, color="violet", opacity=0.8),
                name="halo",
            )
        )
        vel_yz.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel YZ",
            xaxis_title="y [kpc]",
            yaxis_title="z [kpc]",
        )
        vel_yz.show()
    elif proj == "xz":
        x_i = lmc[0, 0]
        z_i = lmc[0, 2]
        vel_xz = ff.create_quiver(
            halo1[:, 0],
            halo1[:, 2],
            halo1[:, 3],
            halo1[:, 5],
            arrow_scale=0.25,
            name="vel",
            marker=dict(color="orange", size=5),
            line=dict(width=1),
        )
        vel_xz.add_trace(
            go.Scatter(
                x=lmc[:, 0],
                y=lmc[:, 2],
                mode="lines",
                marker=dict(size=2, color="blue", opacity=0.8),
                name="lmc",
            )
        )
        vel_xz.add_trace(
            go.Scatter(
                x=[x_i],
                y=[z_i],
                mode="markers",
                marker=dict(size=10, color="green", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_xz.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 2],
                mode="markers",
                marker=dict(size=2, color="violet", opacity=0.8),
                name="halo",
            )
        )
        vel_xz.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XZ",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        vel_xz.show()
    elif proj == "xy":
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy = ff.create_quiver(
            halo1[:, 0],
            halo1[:, 1],
            halo1[:, 3],
            halo1[:, 4],
            arrow_scale=0.25,
            name="vel",
            marker=dict(color="orange", size=5),
            line=dict(width=1),
        )
        # vel_xy.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 1], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xy.add_trace(
            go.Scatter(
                x=[x_i],
                y=[y_i],
                mode="markers",
                marker=dict(size=10, color="green", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_xy.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 1],
                mode="markers",
                marker=dict(size=2, color="violet", opacity=0.8),
                name="halo",
            )
        )
        vel_xy.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XY",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
        )
        vel_xy.show()


def trajectory_LMC(lmc, halo, slice_size, proj):
    """
    :param halo:
    :param lmc:
    :param proj:
    :param slice_size:
    :return: vector field of the halo and lmc in the given projection
    """
    if slice_size > halo.shape[0]:
        print(
            "slice size is larger than the halo size, setting slice size to halo size"
        )
        slice_size = halo.shape[0]
    random_slice = np.random.choice(halo.shape[0], slice_size, replace=False)
    halo1 = halo[random_slice]
    if proj == "yz":
        y_i = lmc[0, 1]
        z_i = lmc[0, 2]
        y_f = lmc[-1, 1]
        z_f = lmc[-1, 2]
        vel_yz = go.Figure()
        # vel_yz = ff.create_quiver(halo1[:,1], halo1[:,2], halo1[:,4], halo1[:,5], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        # vel_yz.add_trace(go.Scatter(x=lmc[:, 1], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_yz.add_trace(
            go.Scatter(
                x=[y_i],
                y=[z_i],
                mode="markers",
                marker=dict(size=10, color="orange", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_yz.add_trace(
            go.Scatter(
                x=[y_f],
                y=[z_f],
                mode="markers",
                marker=dict(size=10, color="purple", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_yz.add_trace(
            go.Scatter(
                x=halo[:, 1],
                y=halo[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.8),
                name="halo",
            )
        )
        vel_yz.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="orbita LMC en YZ",
            xaxis_title="y [kpc]",
            yaxis_title="z [kpc]",
        )
        vel_yz.show()
    elif proj == "xz":
        x_i = lmc[0, 0]
        z_i = lmc[0, 2]
        x_f = lmc[-1, 0]
        z_f = lmc[-1, 2]
        # vel_xz = ff.create_quiver(halo1[:,0], halo1[:,2])
        vel_xz = go.Figure()
        # vel_xz.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xz.add_trace(
            go.Scatter(
                x=[x_i],
                y=[z_i],
                mode="markers",
                marker=dict(size=10, color="orange", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_xz.add_trace(
            go.Scatter(
                x=[x_f],
                y=[z_f],
                mode="markers",
                marker=dict(size=10, color="purple", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_xz.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.8),
                name="halo",
            )
        )
        vel_xz.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="Orbita LMC en XZ",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        vel_xz.show()
    elif proj == "xy":
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy = ff.create_quiver(
            halo1[:, 0],
            halo1[:, 1],
            halo1[:, 3],
            halo1[:, 4],
            arrow_scale=0.25,
            name="vel",
            marker=dict(color="orange", size=5),
            line=dict(width=1),
        )
        # vel_xy.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 1], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xy.add_trace(
            go.Scatter(
                x=[x_i],
                y=[y_i],
                mode="markers",
                marker=dict(size=10, color="orange", opacity=0.8),
                name="lmc_(0,0)",
            )
        )
        vel_xy.add_trace(
            go.Scatter(
                x=halo[:, 0],
                y=halo[:, 1],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.8),
                name="halo",
            )
        )
        vel_xy.update_layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="Orbita LMC en XY",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
        )
        vel_xy.show()


def vec_field2(pert, no_pert, lmc, proj):
    if proj == "yz":
        vel_yz = ff.create_quiver(
            no_pert[:, 1],
            no_pert[:, 2],
            (pert[:, 1] - no_pert[:, 1]),
            (pert[:, 2] - no_pert[:, 2]),
            arrow_scale=0.1,
            name="no pert",
            line=dict(width=0.5),
            marker=dict(size=2),
        )
        x_i = lmc[0, 1]
        y_i = lmc[0, 2]
        vel_yz.add_trace(
            go.Scatter(
                x=pert[:, 1],
                y=pert[:, 2],
                marker=dict(size=1, color="red", opacity=0.8),
                mode="markers",
                name="pert",
            )
        )
        vel_yz.add_trace(
            go.Scatter(
                x=lmc[:, 1],
                y=lmc[:, 2],
                mode="lines",
                marker=dict(size=1, color="green", opacity=0.8),
                name="lmc xy",
            )
        )
        vel_yz.update_layout(
            title="vel wake  YZ",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
            autosize=False,
            width=800,
            height=800,
        )
        vel_yz.add_trace(
            go.Scatter(
                x=[x_i],
                y=[y_i],
                mode="markers",
                marker=dict(size=10, color="blue", opacity=0.8),
                name="lmc_i",
            )
        )
        vel_yz.show()
    elif proj == "xz":
        vel_xz = ff.create_quiver(
            no_pert[:, 0],
            no_pert[:, 2],
            (pert[:, 0] - no_pert[:, 0]),
            (pert[:, 2] - no_pert[:, 2]),
            arrow_scale=0.1,
            name="no pert",
            line=dict(width=0.5),
            marker=dict(size=2),
        )
        x_i = lmc[0, 0]
        y_i = lmc[0, 2]
        vel_xz.add_trace(
            go.Scatter(
                x=pert[:, 0],
                y=pert[:, 2],
                marker=dict(size=1, color="red", opacity=0.8),
                mode="markers",
                name="pert",
            )
        )
        vel_xz.add_trace(
            go.Scatter(
                x=lmc[:, 0],
                y=lmc[:, 2],
                mode="lines",
                marker=dict(size=1, color="green", opacity=0.8),
                name="lmc xy",
            )
        )
        vel_xz.update_layout(
            title="vel wake  XZ",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
            autosize=False,
            width=800,
            height=800,
        )
        vel_xz.add_trace(
            go.Scatter(
                x=[x_i],
                y=[y_i],
                mode="markers",
                marker=dict(size=10, color="blue", opacity=0.8),
                name="lmc_i",
            )
        )
        vel_xz.show()
    elif proj == "xy":
        vel_xy = ff.create_quiver(
            no_pert[:, 0],
            no_pert[:, 1],
            (pert[:, 0] - no_pert[:, 0]),
            (pert[:, 1] - no_pert[:, 1]),
            arrow_scale=0.1,
            name="no pert",
            line=dict(width=0.5),
            marker=dict(size=2),
        )
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy.add_trace(
            go.Scatter(
                x=pert[:, 0],
                y=pert[:, 1],
                marker=dict(size=1, color="red", opacity=0.8),
                mode="markers",
                name="pert",
            )
        )
        vel_xy.add_trace(
            go.Scatter(
                x=lmc[:, 0],
                y=lmc[:, 1],
                mode="lines",
                marker=dict(size=1, color="green", opacity=0.8),
                name="lmc xy",
            )
        )
        vel_xy.update_layout(
            title="vel wake  XY",
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
            autosize=False,
            width=800,
            height=800,
        )
        vel_xy.add_trace(
            go.Scatter(
                x=[x_i],
                y=[y_i],
                mode="markers",
                marker=dict(size=10, color="blue", opacity=0.8),
                name="lmc_i",
            )
        )
        vel_xy.show()


def E_L(halo, proj, type="hist", halo2=None, slice=None, x_limits=None, y_limits=None):
    if slice is None:
        size = min(halo.shape[0], halo2.shape[0])
    else:
        size = min(slice, halo.shape[0], halo2.shape[0])
    random_slice = np.random.choice(halo.shape[0], size, replace=False)
    if halo2 is not None:
        random_slice2 = np.random.choice(halo2.shape[0], size, replace=False)
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111)
    if type == "hist":
        if halo2 is not None:
            if proj == "x":
                x_data = halo[:, 11][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 11][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                hist1, xedges1, yedges1, im1 = plt.hist2d(
                    x_data, y_data, bins=30, cmap="autumn_r", alpha=0.5
                )
                hist2, xedges2, yedges2, im2 = plt.hist2d(
                    x2_data, y2_data, bins=30, cmap="winter_r", alpha=0.5
                )
                plt.colorbar(im1, label="{halo}")
                plt.colorbar(im2, label="{halo2}")
                plt.xlabel("Momento angular L_x (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_x density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "y":
                x_data = halo[:, 12][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 12][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                hist1, xedges1, yedges1, im1 = plt.hist2d(
                    x_data, y_data, bins=30, cmap="autumn_r", alpha=0.5
                )
                hist2, xedges2, yedges2, im2 = plt.hist2d(
                    x2_data, y2_data, bins=30, cmap="winter_r", alpha=0.5
                )
                plt.colorbar(im1, label="{halo}")
                plt.colorbar(im2, label="{halo2}")
                plt.xlabel("Momento angular L_y (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_y Density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "z":
                x_data = halo[:, 13][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 13][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                hist1, xedges1, yedges1, im1 = plt.hist2d(
                    x_data, y_data, bins=30, cmap="autumn_r", alpha=0.5
                )
                hist2, xedges2, yedges2, im2 = plt.hist2d(
                    x2_data, y2_data, bins=30, cmap="winter_r", alpha=0.5
                )
                plt.colorbar(im1, label="{halo}")
                plt.colorbar(im2, label="{halo2}")
                plt.xlabel("Momento angular L_z (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_z Density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "mag":
                x_data = halo[:, 10][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 10][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                hist1, xedges1, yedges1, im1 = plt.hist2d(
                    x_data, y_data, bins=30, cmap="autumn_r", alpha=0.5
                )
                hist2, xedges2, yedges2, im2 = plt.hist2d(
                    x2_data, y2_data, bins=30, cmap="winter_r", alpha=0.5
                )
                plt.colorbar(im1, label="{halo}")
                plt.colorbar(im2, label="{halo2}")
                plt.xlabel("Momento angular L_mag (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_mag Density Z")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
        elif halo2 is None:
            if proj == "x":
                x_data = halo[:, 11][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.hist2d(x_data, y_data, bins=30, cmap="viridis")
                plt.colorbar(label="Particle Density")
                plt.xlabel("Momento angular L_x (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_x density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "y":
                x_data = halo[:, 12][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.hist2d(x_data, y_data, bins=30, cmap="viridis")
                plt.colorbar(label="Particle Density")
                plt.xlabel("Momento angular L_y (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_y Density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "z":
                x_data = halo[:, 13][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.hist2d(x_data, y_data, bins=30, cmap="viridis")
                plt.colorbar(label="Particle Density")
                plt.xlabel("Momento angular L_z (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_z Density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "mag":
                x_data = halo[:, 10][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.hist2d(x_data, y_data, bins=30, cmap="viridis")
                plt.colorbar(label="Particle Density")
                plt.xlabel("Momento angular L_mag (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_mag Density Z")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
    elif type == "scatter":
        if halo2 is None:
            if proj == "x":
                x_data = halo[:, 11][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.scatter(x_data, y_data, s=1)
                plt.xlabel("Momento angular L_x (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_x density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "y":
                x_data = halo[:, 12][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.scatter(x_data, y_data, s=1)
                plt.xlabel("Momento angular L_y (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("2D Histogram of L_y Density vs E")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "z":
                x_data = halo[:, 13][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.scatter(x_data, y_data, s=1)
                plt.xlabel("Momento angular L_z (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("E(L_z)")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "mag":
                x_data = halo[:, 10][random_slice]
                y_data = halo[:, 15][random_slice]
                plt.scatter(x_data, y_data, s=1)
                plt.xlabel("Momento angular L_mag (kpc km/s)")
                plt.ylabel("Energia (km^2/s^2)")
                plt.title("E(L_mag) ")
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
        else:
            if proj == "x":
                x_data = halo[:, 11][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 11][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                ax.scatter(x_data, y_data, s=1, c="#1E90FF", label="Unperturbed")
                ax.scatter(x2_data, y2_data, s=1, c="#DC143C", label="Perturbed")
                ax.set_xlabel(r'Momento angular $L_x$ (kpc km/s)', fontsize=14)
                ax.set_ylabel(r"Energia ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)", fontsize=14)
                ax.set_title(r'Phase diagram $E$ vs $L_x$', fontsize=20)
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                # set aspect ratio to be equal
                ax.set_aspect('equal', adjustable='box')
                plt.show()
            elif proj == "y":
                x_data = halo[:, 12][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 12][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                ax.scatter(x_data, y_data, s=1, c="#1E90FF", label="Unperturbed")
                ax.scatter(x2_data, y2_data, s=1, c="#DC143C", label="Perturbed")
                ax.set_xlabel(r'Momento angular $L_y$ (kpc km/s)', fontsize=14)
                ax.set_ylabel(r'Energia ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)', fontsize=14)
                ax.set_title(r'Phase diagram $E$ vs $L_y$', fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "z":
                x_data = halo[:, 13][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 13][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                ax.scatter(x_data, y_data, s=1, c="#1E90FF", label="Unperturbed")
                ax.scatter(x2_data, y2_data, s=1, c="#DC143C", label="Perturbed")
                ax.set_xlabel(r'Momento angular $L_z$ (kpc km/s)',fontsize=14)
                ax.set_ylabel(r'Energia ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)', fontsize=14)
                ax.set_title(r'Phase diagram $E$ vs $L_z$', fontsize=20)                
                ax.set_aspect('equal', adjustable='box')
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
            elif proj == "mag":
                x_data = halo[:, 10][random_slice]
                y_data = halo[:, 15][random_slice]
                x2_data = halo2[:, 10][random_slice2]
                y2_data = halo2[:, 15][random_slice2]
                ax.scatter(x_data, y_data, s=1, c="#1E90FF", label="Unperturbed")
                ax.scatter(x2_data, y2_data, s=1, c="#DC143C", label="Perturbed")
                ax.set_xlabel(r'Momento angular $L_{mag}$ (kpc km/s)',fontsize=14)
                ax.set_ylabel(r'Energia ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)',fontsize=14)
                ax.set_title(r'Phase diagram $E$ vs $L_{mag}$', fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                if x_limits is not None:
                    plt.xlim(x_limits)
                if y_limits is not None:
                    plt.ylim(y_limits)
                plt.show()
    else:
        print("No se puede graficar, no hay datos")


def comparison_E_L(halo1, halo2, halo3, halo4, proj, slice=None):
    """
    This function creates a side-by-side comparison of two sets of halo data.
    Each subplot contains two halos, with the first subplot showing the perturbed halos
    and the second subplot showing the unperturbed halos.

    Parameters:
    halo1, halo2, halo3, halo4: numpy arrays containing the halo data
    proj: string indicating the projection to use ('x', 'y', 'z', or 'mag')
    slice: integer indicating the number of data points to use (default is None, which uses all data points)
    """
    # Ensure that the 'size' variable is not larger than the size of any halo dataset
    if slice is None:
        size = min(halo1.shape[0], halo2.shape[0], halo3.shape[0], halo4.shape[0])
    else:
        size = min(slice, halo1.shape[0], halo2.shape[0], halo3.shape[0], halo4.shape[0])
    # Create random slices for each halo
    random_slices = [np.random.choice(halo.shape[0], size, replace=False) for halo in [halo1, halo2, halo3, halo4]]
    # Define the indices for the data based on the projection
    proj_indices = {'x': (11, 15), 'y': (12, 15), 'z': (13, 15), 'mag': (10, 15)}
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=300)
    # Iterate over the halos and subplots
    for i, (ax, halo1_slice, halo2_slice) in enumerate(zip(axes, random_slices[::2], random_slices[1::2])):
        x1_data, y1_data = halo1[:, proj_indices[proj][0]][halo1_slice], halo1[:, proj_indices[proj][1]][halo1_slice]
        x2_data, y2_data = halo2[:, proj_indices[proj][0]][halo2_slice], halo2[:, proj_indices[proj][1]][halo2_slice]
        ax.scatter(x1_data, y1_data, s=1, c="#1E90FF", label=f"Halo {2*i+1}")
        ax.scatter(x2_data, y2_data, s=1, c="#DC143C", label=f"Halo {2*i+2}")
        ax.set_xlabel(r'Momento angular $L_x$ (kpc km/s)', fontsize=14)
        ax.set_ylabel(r"Energia ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)", fontsize=14)
        ax.set_title(f'Phase diagram $E$ vs $L_x$ ({"Perturbed" if i == 0 else "Unperturbed"} halos)', fontsize=20)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
    plt.show()


def E_L_contour(halo, proj):
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111)
    if proj == "x":
        x_data = halo[:, 11]
        y_data = halo[:, 15]
        
        ax.set_xlabel(r'Specific angular momentum $L_x$ (kpc km/s)', fontsize=14)
        ax.set_title(r'Phase diagram $E$ vs $L_x$', fontsize=20)
    elif proj == "y":
        x_data = halo[:, 12]
        y_data = halo[:, 15]

        ax.set_xlabel(r'Specific angular momentum $L_y$ (kpc km/s)', fontsize=14)
        ax.set_title(r'Phase diagram $E$ vs $L_y$', fontsize=20)
    elif proj == "z":
        x_data = halo[:, 13]
        y_data = halo[:, 15]
        
        ax.set_xlabel(r'Specific angular momentum $L_z$ (kpc km/s)', fontsize=14)
        ax.set_title(r'Phase diagram $E$ vs $L_z$', fontsize=20)
    elif proj == "mag":
        x_data = halo[:, 10]
        y_data = halo[:, 15]
        
        ax.set_xlabel(r'Magnitude of specific angular momentum $L$ (kpc km/s)', fontsize=14)
        ax.set_title(r'Phase diagram $E$ vs $L$', fontsize=20)
    else:
        raise ValueError("Invalid projection")
        
    # Create 2D histograms and filled contour plots for halo
    
    hist1, xedges1, yedges1 = np.histogram2d(x_data, y_data, bins=50)
    X1, Y1 = np.meshgrid(xedges1[:-1], yedges1[:-1])
    cf1 = ax.contourf(X1, Y1, hist1.T, cmap='Blues', alpha=0.7)

    # Add colorbar for contour plot
    cbar1 = fig.colorbar(cf1, ax=ax, shrink=0.5, aspect=10, pad=0.02)
    cbar1.set_label('Unperturbed Halo Density', fontsize=12)
    
    ax.set_ylabel(r"Energy ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)", fontsize=14)
    ax.set_aspect('equal', adjustable='box')

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def econt_side_by_side(halo1, halo2, proj):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), dpi=300, sharey=True)
    fig.subplots_adjust(right=0.8)  # Make room for the colorbar

    # Calculate global minimum and maximum density values for both halos
    hist1, _, _ = np.histogram2d(halo1[:, 11], halo1[:, 15], bins=50)
    hist2, _, _ = np.histogram2d(halo2[:, 11], halo2[:, 15], bins=50)
    global_vmin = min(hist1.min(), hist2.min())
    global_vmax = max(hist1.max(), hist2.max())

    for halo, ax, title in zip([halo1, halo2], [ax1, ax2], ['Unperturbed halo', 'Perturbed halo']):
        if proj == "x":
            x_data = halo[:, 11]
            y_data = halo[:, 15]
            ax.set_xlabel(r'Specific angular momentum $L_x$ (kpc km/s)', fontsize=14)
        elif proj == "y":
            x_data = halo[:, 12]
            y_data = halo[:, 15]
            ax.set_xlabel(r'Specific angular momentum $L_y$ (kpc km/s)', fontsize=14)
        elif proj == "z":
            x_data = halo[:, 13]
            y_data = halo[:, 15]
            ax.set_xlabel(r'Specific angular momentum $L_z$ (kpc km/s)', fontsize=14)
        elif proj == "mag":
            x_data = halo[:, 10]
            y_data = halo[:, 15]
            ax.set_xlabel(r'Magnitude of specific angular momentum $L$ (kpc km/s)', fontsize=14)
        else:
            raise ValueError("Invalid projection")

        # Create 2D histograms and filled contour plots for halo
        hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=50)
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        cf = ax.contourf(X, Y, hist.T, cmap='Blues', norm=Normalize(vmin=global_vmin, vmax=global_vmax))
        
        ax.set_ylabel(r"Energy ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)", fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=24)  # Add subplot title
        # Add a vertical line at x=0 for each subplot
        ax.axvline(x=10000, color='orange', linestyle='--', linewidth=1)
        ax.axvline(x=-15000, color='orange', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='#32CD32', linestyle='--', linewidth=1)
        
        # ax.axvline(y=, color='orange', linestyle='--', linewidth=1)
        
    # Create a ScalarMappable object with the same colormap and normalization as the contour plots
    sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])  # Dummy array for the ScalarMappable

    # Add a single colorbar for both contour plots
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Halo Density', shrink=0.5)

    fig.suptitle('Phase diagram comparison', fontsize=20)  # Add general title for the whole figure
    plt.show()
    
    
    
    
def econt_scatter_side(halo1, halo2, halo3, halo4, proj):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), dpi=300, sharey=True)
    fig.subplots_adjust(right=0.8)  # Make room for the colorbar

    # Calculate global minimum and maximum density values for both halos
    hist1, _, _ = np.histogram2d(halo1[:, 11], halo1[:, 15], bins=50)
    hist2, _, _ = np.histogram2d(halo2[:, 11], halo2[:, 15], bins=50)
    global_vmin = min(hist1.min(), hist2.min())
    global_vmax = max(hist1.max(), hist2.max())

    for halo, ax, title, selected_halo in zip([halo1, halo2], [ax1, ax2], ['Unperturbed halo', 'Perturbed halo'], [halo3, halo4]):
        if proj == "x":
            x_data = halo[:, 11]
            y_data = halo[:, 15]
            x_sel = 11
            ax.set_xlabel(r'Specific angular momentum $L_x$ (kpc km/s)', fontsize=14)
        elif proj == "y":
            x_data = halo[:, 12]
            y_data = halo[:, 15]
            x_sel = 12
            ax.set_xlabel(r'Specific angular momentum $L_y$ (kpc km/s)', fontsize=14)
        elif proj == "z":
            x_data = halo[:, 13]
            y_data = halo[:, 15]
            x_sel = 13
            ax.set_xlabel(r'Specific angular momentum $L_z$ (kpc km/s)', fontsize=14)
        elif proj == "mag":
            x_data = halo[:, 10]
            y_data = halo[:, 15]
            x_sel = 10
            ax.set_xlabel(r'Magnitude of specific angular momentum $L$ (kpc km/s)', fontsize=14)
        else:
            raise ValueError("Invalid projection")

        # Create 2D histograms and filled contour plots for halo
        hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=50)
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        cf = ax.contourf(X, Y, hist.T, cmap='Blues', norm=Normalize(vmin=global_vmin, vmax=global_vmax))
        
        ax.scatter(selected_halo[:, x_sel], selected_halo[:,15], s=1, color='#DC143C', label='Selected halo', alpha = 0.5)
        ax.set_ylabel(r"Energy ($\frac{\mathrm{km}^2}{\mathrm{s}^2}$)", fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=24)  # Add subplot title
        # Add a vertical line at x=0 for each subplot
        ax.axvline(x=10000, color='orange', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='#32CD32', linestyle='--', linewidth=1)
        ax.axvline(x=15000, color='purple', linestyle='--', linewidth=1)
        # ax.axvline(y=, color='orange', linestyle='--', linewidth=1)
        
    # Create a ScalarMappable object with the same colormap and normalization as the contour plots
    sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])  # Dummy array for the ScalarMappable

    # Add a single colorbar for both contour plots
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Halo Density', shrink=0.5)

    fig.suptitle('Phase diagram comparison', fontsize=20)  # Add general title for the whole figure
    plt.show()
    
def sel3(sel_3, rel_lmc, proj):
    if proj == "xy":
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(
            go.Scatter(
                x=sel_3[:, 0],
                y=sel_3[:, 1],
                mode="markers",
                marker=dict(
                    size=3,
                    color=sel_3[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="wake xy",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[:, 0],
                y=rel_lmc[:, 1],
                mode="lines",
                marker=dict(
                    size=2,
                    color="blue",
                    opacity=0.8,
                ),
                name="LMC xy",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[-1:, 0],
                y=rel_lmc[-1:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color="purple",
                    opacity=0.8,
                ),
                name="(x_f,y_f)",
            )
        )
        fig_sel3.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XY",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        fig_sel3.show()
    elif proj == "xz":
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(
            go.Scatter(
                x=sel_3[:, 1],
                y=sel_3[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=sel_3[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="wake yz",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[:, 1],
                y=rel_lmc[:, 2],
                mode="lines",
                marker=dict(
                    size=2,
                    color="blue",
                    opacity=0.8,
                ),
                name="lmc yz",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[-1:, 1],
                y=rel_lmc[-1:, 2],
                mode="markers",
                marker=dict(
                    size=10,
                    color="purple",
                    opacity=0.8,
                ),
                name="(y_f,z_f)",
            )
        )
        fig_sel3.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel YZ",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        fig_sel3.show()
    elif proj == "yz":
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(
            go.Scatter(
                x=sel_3[:, 0],
                y=sel_3[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=sel_3[:, 10],
                    colorscale="viridis",
                    opacity=0.8,
                    colorbar=dict(
                        title="L [kpc km/s]",
                        titleside="top",
                        tickmode="array",
                        tickvals=[10000, 20000, 30000],
                        ticktext=["10000", "20000", "30000"],
                        ticks="outside",
                        thickness=20,
                        len=0.5,
                        y=0.5,
                        yanchor="middle",
                        x=1.1,
                        xanchor="left",
                    ),
                ),
                name="wake xz",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[:, 0],
                y=rel_lmc[:, 2],
                mode="lines",
                marker=dict(
                    size=2,
                    color="blue",
                    opacity=0.8,
                ),
                name="lmc xz",
            )
        )
        fig_sel3.add_trace(
            go.Scatter(
                x=rel_lmc[-1:, 0],
                y=rel_lmc[-1:, 2],
                mode="markers",
                marker=dict(
                    size=10,
                    color="purple",
                    opacity=0.8,
                ),
                name="(x_f,z_f)",
            )
        )
        fig_sel3.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            title="DM wake sel XZ",
            xaxis_title="x [kpc]",
            yaxis_title="z [kpc]",
        )
        fig_sel3.show()
