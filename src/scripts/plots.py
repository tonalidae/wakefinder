import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
def hist_r_l(halo):
    """ Plot the histogram of the position of the LMC around the MW.

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
    ax.set_ylabel(r'Magnitud momento inicial  [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]', fontsize=8)
    ax.set_xlabel('Distancia desde el centro del halo [kpc]', fontsize=10)
    ax.set_ylabel(r'Magnitud momento angular inicial [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]', fontsize=10)
    plt.suptitle("L mag vs pos mag halo perturbado", fontsize=15)
    im = plt.imshow(np.log10(hist_r_L_rot.T), extent=[50, 250, 0, 50000], aspect='auto', origin='lower', vmin=0,
                    vmax=5, cmap='Greys')
    plt.tight_layout()
    plt.show()
    return im
    # plt.savefig('./../media/imgs/wakefinder/' + 'Lmag_pos' + '.png', bbox_inches='tight', dpi=400)

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
    ax.set_ylabel(r'Momento angular x  [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]', fontsize=8)
    ax.set_xlabel('Distancia desde el centro del halo [kpc]', fontsize=10)
    ax.set_ylabel(r'momento angular inicial [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]', fontsize=10)
    plt.suptitle("L_x vs R_x halo perturbado", fontsize=15)
    im = plt.imshow(np.log10(hist_r_L_rot.T), extent=[50, 250, 0, 50000], aspect='auto', origin='lower', vmin=0,
                    vmax=5, cmap='Greys')
    plt.tight_layout()
    plt.show()
    return im



def hist_L_r_low(pos,ang_m):
    r =  np.arange(50,200,10)
    L = np.arange(9,50000,2500)
    hist_r_L = np.zeros((len(r),len(L)))
    for i in range(0,len(r)-1):
        r_index = np.where((pos > r[i]) & (pos < r[i+1]))
        for j in range(0,len(L)-1):
            Nl = np.where((ang_m[r_index]> L[j]) & (ang_m[r_index] < L[j+1]))
            hist_r_L[i,j] = np.shape(Nl)[1]
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    im = plt.imshow(np.log10(hist_r_L.T), extent=[50, 195, 0, 50000], aspect='auto', origin='lower', vmin=0, vmax=5,cmap='Greys')
    ax.set_xlabel(r'distancias en [$\mathrm{kpc}$]', fontsize=8)
    ax.set_ylabel(r'Momhisento angular inicial [$ \mathrm{kpc} \mathrm{ km} \mathrm{ s^{-1}}$]', fontsize=8)
    ax.set_title("Partículas con momento angular inicial bajo", fontsize=12, y=1.05)
    # plt.savefig('../../media/imgs/wakefinder/'+figname+'.png', bbox_inches='tight', dpi = 400)
    plt.show()
    return hist_L_r_low


def surface_plot(orb_pos, n_vector):
    """ Plot the orbit of the LMC around the MW.

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
    fig = go.Figure(data=[go.Surface(z=Z_mesh, x=X_mesh, y=Y_mesh, colorscale='Viridis', opacity=0.5)])
    fig.add_trace(
        go.Scatter3d(x=orb_pos[:, 0], y=orb_pos[:, 1], z=orb_pos[:, 2],
                     mode='markers', marker=dict(size=1, color='#ccc')))
    # fig.add_trace(
    #     go.Scatter3d(x=[0, normal_vector_line[0]], y=[0, normal_vector_line[1]], z=[0, normal_vector_line[2]],
    #                  mode='lines',
    #                  line=dict(width=5)))
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))
    return fig

def LMC_plot(orb_pos, proj):
    """ Plot the orbit of the LMC around the MW.

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
        LMC_xy = go.Figure(data=[go.Line(x=orb_pos[:, 0], y=orb_pos[:, 1], name='LMC orbit rot xy')])
        LMC_xy.add_trace(go.Scatter(x=[initial_point[0]], y=[initial_point[1]], mode='markers', name='initial point', marker=dict(size=10, color='orange')))
        LMC_xy.add_trace(go.Scatter(x=[last_point[0]], y=[last_point[1]], mode='markers', name='last point', marker=dict(size=10, color='purple')))
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
            title_xanchor='center',
            title_yanchor='top',
        )
        LMC_xy.update_xaxes(title_text="X [kpc]", title_font_size=15)
        LMC_xy.update_yaxes(title_text="Y [kpc]", title_font_size=15)
        LMC_xy.show()
    elif proj == "xz":
        initial_point = orb_pos[0, 0:3]
        last_point = orb_pos[-1, 0:3]
        v_0 = orb_pos[0, 3:6]
        v_f = orb_pos[-1, 3:6]
        LMC_xz = go.Figure(data=[go.Line(x=orb_pos[:, 0], y=orb_pos[:, 2], name='LMC orbit rot xz')])
        LMC_xz.add_trace(go.Scatter(x=[initial_point[0]], y=[initial_point[2]], mode='markers', name='initial point', marker=dict(size=10, color='orange')))
        LMC_xz.add_trace(go.Scatter(x=[last_point[0]], y=[last_point[2]], mode='markers', name='last point', marker=dict(size=10, color='purple')))
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
            title_xanchor='center',
            title_yanchor='top',
        )
        LMC_xz.update_xaxes(title_text="X [kpc]", title_font_size=15)
        LMC_xz.update_yaxes(title_text="Z [kpc]", title_font_size=15)
        LMC_xz.show()
    elif proj == "yz":
        initial_point = orb_pos[0, 0:3]
        last_point = orb_pos[-1, 0:3]
        velocity = orb_pos[0, 3:6]
        LMC_yz = go.Figure(data=[go.Line(x=orb_pos[:, 1], y=orb_pos[:, 2], name='LMC orbit rot yz')])
        LMC_yz.add_trace(go.Scatter(x=[initial_point[1]], y=[initial_point[2]], mode='markers', name='initial point', marker=dict(size=10, color='orange')))
        LMC_yz.add_trace(go.Scatter(x=[last_point[1]], y=[last_point[2]], mode='markers', name='last point', marker=dict(size=10, color='purple')))
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
            title_xanchor='center',
            title_yanchor='top',
        )
        LMC_yz.update_xaxes(title_text="Y [kpc]", title_font_size=15)
        LMC_yz.update_yaxes(title_text="Z [kpc]", title_font_size=15)
        LMC_yz.show()


def hist(halo, orb, proj):
    """ Plot the histogram of the distances between the LMC and the MW.

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
        fig.add_trace(go.Scatter(x=halo[:, 0],
                                 y=halo[:, 1],
                                 mode='markers',
                                 marker=dict(color='deepskyblue', size=1),
                                 name='MW DM Halo particles'))

        # Create line plot
        fig.add_trace(go.Scatter(x=orb[:, 0],
                                 y=orb[:, 1],
                                 mode='lines',
                                 line=dict(color='chartreuse', width=1),
                                 name='LMC Orbit'))

        # Set axis labels and title
        fig.update_layout(xaxis_title=r'x [$\mathrm{kpc}$]', yaxis_title=r'y [$\mathrm{kpc}$]',
                          title='Proyeccion xy', title_font_size=14)

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
            )
        )
        # fig.write_image("proj_xy.png")
    elif proj == "xz":
        fig.add_trace(go.Scatter(x=halo[:, 0],
                                    y=halo[:, 2],
                                    mode='markers',
                                    marker=dict(color='deepskyblue', size=1),
                                    name='MW DM Halo particles'))
        fig.add_trace(go.Scatter(x=orb[:, 0],
                                    y=orb[:, 2],
                                    mode='lines',
                                    line=dict(color='chartreuse', width=1),
                                    name='LMC Orbit'))
        fig.update_layout(xaxis_title=r'x [$\mathrm{kpc}$]', yaxis_title=r'z [$\mathrm{kpc}$]',
                            title='Proyeccion xz', title_font_size=14)
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
            )
        )
        print("paso por aqui")
        # fig.write_image("proj_xz.png")
    elif proj == "yz":
        fig.add_trace(go.Scatter(x=halo[:, 1],
                                    y=halo[:, 2],
                                    mode='markers',
                                    marker=dict(color='deepskyblue', size=1),
                                    name='MW DM Halo particles'))
        fig.add_trace(go.Scatter(x=orb[:, 1],
                                    y=orb[:, 2],
                                    mode='lines',
                                    line=dict(color='chartreuse', width=1),
                                    name='LMC Orbit'))
        fig.update_layout(xaxis_title=r'y [$\mathrm{kpc}$]', yaxis_title=r'z [$\mathrm{kpc}$]',
                            title='Proyeccion yz', title_font_size=14)
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
            )
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

def hist_delta_xy(pos,pos_oy,pos_oz,index_percentage):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    plt.scatter(pos[index_percentage,0], pos[index_percentage,1], s=5, alpha=0.5, c=['deepskyblue'], marker='.')
    plt.plot(pos_oy, pos_oz, alpha=0.8, c='chartreuse', linewidth=1.5)
    ax.set_xlabel(r'x [$\mathrm{kpc}$]', fontsize=20)
    ax.set_ylabel(r'y [$\mathrm{kpc}$]', fontsize=20)
    plt.xlim(-150,150)
    plt.ylim(-150,150)
    # plt.title(r"Projection $xy$ high $L_$ ", fontsize=12)
    # ax.grid(which='both')
    return hist_delta_xy

def plot_3d(arr1, arr2, title):
    plot_3d = go.Figure()
    plot_3d.add_trace(go.Scatter3d(x=arr1[:, 0], y=arr1[:, 1], z=arr1[:, 2], mode='markers', marker=dict(size=1, color=arr1[:,10], colorscale='viridis', opacity=0.8), name='wake'))
    plot_3d.add_trace(
    go.Scatter3d(x=arr2[:, 0], y=arr2[:, 1], z=arr2[:, 2], mode='lines', marker=dict(size=1, color='red', opacity=0.8),
            name='lmc'))
    plot_3d.update_layout(
    scene=dict(xaxis_title='x [kpc]', yaxis_title='y [kpc]', zaxis_title='z [kpc]', aspectratio=dict(x=1, y=1, z=1)),
    title= title, title_font_size=14)
    plot_3d.add_trace(go.Cone(x=[arr2[-1, 0]], y=[arr2[-1, 1]], z = [arr2[-1, 2]],
                    u=[arr2[-1, 3] - arr2[-2, 3]], v=[arr2[-1, 4] - arr2[-2, 4]], w = [arr2[-1,5]-arr2[-2,-5]] ,sizemode="absolute", sizeref=10
                    , anchor="tail", showscale=False,  opacity=0.8, name='lmc velocity'))
    plot_3d.show()

def proj_both(arr1, arr2, arr3, proj):
    if proj == "xy":
        proj_xy_both = go.Figure()
        proj_xy_both.add_trace(go.Scatter(x=arr1[:, 0], y=arr1[:, 1], mode='markers',
                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                    colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=20,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.1,
                                    xanchor='left'
                                    )
                                    ), name='halo pert'))
        proj_xy_both.add_trace(go.Scatter(x=arr2[:, 0], y=arr2[:, 1], mode='markers',
                    marker=dict(size=2, color=arr2[:, 10], colorscale='agsunset', opacity=0.8,
                                colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=20,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.1,
                                    xanchor='left'
                                    )), name='halo_no_pert'))
        proj_xy_both.add_trace(go.Scatter(x=arr3[:, 0], y=arr3[:, 1], mode='lines', marker=dict(size=1, color='blue', opacity=0.8),
                                name='lmc xy'))
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
            title='DM wake sel XY',
            xaxis_title='x [kpc]',
            yaxis_title='y [kpc]'
        )
        proj_xy_both.show()
    elif proj == "xz":
        proj_xz_both = go.Figure()
        proj_xz_both.add_trace(go.Scatter(x=arr1[:, 0], y=arr1[:, 2], mode='markers',
                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                    colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=20,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.1,
                                    xanchor='left'
                                    )
                                    ), name='halo pert'))
        proj_xz_both.add_trace(go.Scatter(x=arr2[:, 0], y=arr2[:, 2], mode='markers',
                    marker=dict(size=2, color=arr2[:, 10], colorscale='agsunset', opacity=0.8,
                                colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=20,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.1,
                                    xanchor='left'
                                    )), name='halo_no_pert'))
        proj_xz_both.add_trace(go.Scatter(x=arr3[:, 0], y=arr3[:, 2], mode='lines', marker=dict(size=1, color='blue', opacity=0.8),
                                name='lmc xz'))
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
            title='DM wake sel XZ',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        proj_xz_both.show()
    elif proj == "yz":
        proj_yz_both = go.Figure()
        proj_yz_both.add_trace(go.Scatter(x=arr1[:, 1], y=arr1[:, 2], mode='markers',
                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                    colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=5,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.1,
                                    xanchor='left'
                                    )
                                    ), name='halo pert'))
        proj_yz_both.add_trace(go.Scatter(x=arr2[:, 1], y=arr2[:, 2], mode='markers',
                    marker=dict(size=2, color=arr2[:, 10], colorscale='agsunset', colorbar=dict(
                                    title='L [kpc km/s]',
                                    titleside='top',
                                    tickmode='array',
                                    tickvals=[10000, 20000, 30000],
                                    ticktext=['10000', '20000', '30000'],
                                    ticks='outside',
                                    thickness=5,
                                    len=0.5,
                                    y=0.5,
                                    yanchor='middle',
                                    x=1.5,
                                    xanchor='left'
                                    )), name='halo_no_pert'))
        proj_yz_both.add_trace(go.Scatter(x=arr3[:, 1], y=arr3[:, 2], mode='lines', marker=dict(size=1, color='blue', opacity=0.8),
                                name='lmc yz'))
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
            title='DM wake sel YZ',
            xaxis_title='y [kpc]',
            yaxis_title='z [kpc]'
        )
        proj_yz_both.show()
def proj_xy(arr1, arr2):
    proj_xy = go.Figure()
    proj_xy.add_trace(go.Scatter(x=arr1[:, 0], y=arr1[:, 1], mode='markers',
                                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                                colorbar=dict(
                                                    title='L [kpc km/s]',
                                                    titleside='top',
                                                    tickmode='array',
                                                    tickvals=[10000, 20000, 30000],
                                                    ticktext=['10000', '20000', '30000'],
                                                    ticks='outside',
                                                    thickness=20,
                                                    len=0.5,
                                                    y=0.5,
                                                    yanchor='middle',
                                                    x=1.1,
                                                    xanchor='left'
                                                )
                                                ), name='halo pert'))
    proj_xy.add_trace(go.Scatter(x=arr2[:, 0], y=arr2[:, 1], mode='lines', marker=dict(size=1, color='red', opacity=0.8),
                                    name='lmc xy'))
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
        title='DM wake sel XY',
        xaxis_title='x [kpc]',
        yaxis_title='y [kpc]'
    )
    proj_xy.show()

def proj_yz(arr1, arr2):
    proj_yz = go.Figure()
    proj_yz.add_trace(go.Scatter(x=arr1[:, 1], y=arr1[:, 2], mode='markers',
                                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                                colorbar=dict(
                                                    title='L [kpc km/s]',
                                                    titleside='top',
                                                    tickmode='array',
                                                    tickvals=[10000, 20000, 30000],
                                                    ticktext=['10000', '20000', '30000'],
                                                    ticks='outside',
                                                    thickness=20,
                                                    len=0.5,
                                                    y=0.5,
                                                    yanchor='middle',
                                                    x=1.1,
                                                    xanchor='left'
                                                )
                                                ), name='halo pert'))
    proj_yz.add_trace(go.Scatter(x=arr2[:, 1], y=arr2[:, 2], mode='lines', marker=dict(size=1, color='red', opacity=0.8),
                                    name='lmc xy'))
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
        title='DM wake sel YZ',
        xaxis_title='y [kpc]',
        yaxis_title='z [kpc]'
    )
    proj_yz.show()

def proj_xz(arr1, arr2):
    proj_xz = go.Figure()
    proj_xz.add_trace(go.Scatter(x=arr1[:, 0], y=arr1[:, 2], mode='markers',
                                    marker=dict(size=2, color=arr1[:, 10], colorscale='viridis', opacity=0.8,
                                                colorbar=dict(
                                                    title='L [kpc km/s]',
                                                    titleside='top',
                                                    tickmode='array',
                                                    tickvals=[10000, 20000, 30000],
                                                    ticktext=['10000', '20000', '30000'],
                                                    ticks='outside',
                                                    thickness=20,
                                                    len=0.5,
                                                    y=0.5,
                                                    yanchor='middle',
                                                    x=1.1,
                                                    xanchor='left'
                                                )
                                                ), name='halo pert'))

    proj_xz.add_trace(go.Scatter(x=arr2[:, 0], y=arr2[:, 2], mode='lines', marker=dict(size=1, color='red', opacity=0.8),
                                    name='lmc xy'))
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
        title='DM wake sel XZ',
        xaxis_title='x [kpc]',
        yaxis_title='z [kpc]'
    )
    proj_xz.show()

def vector_field(halo, lmc, proj, slice_size):
    '''
    :param halo:
    :param lmc:
    :param proj:
    :param slice_size:
    :return: vector field of the halo and lmc in the given projection
    '''
    if slice_size > halo.shape[0]:
        print('slice size is larger than the halo size, setting slice size to halo size')
        slice_size = halo.shape[0]
    random_slice = np.random.choice(halo.shape[0], slice_size, replace=False)
    halo1 = halo[random_slice]
    if proj=='yz':
        y_i = lmc[0, 1]
        z_i = lmc[0, 2]
        vel_yz = ff.create_quiver(halo1[:,1], halo1[:,2], halo1[:,4], halo1[:,5], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        vel_yz.add_trace(go.Scatter(x=lmc[:, 1], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_yz.add_trace(go.Scatter(x=[y_i], y=[z_i], mode='markers', marker=dict(size=10, color='green', opacity=0.8), name='lmc_(0,0)'))
        vel_yz.add_trace(go.Scatter(x=halo[:, 1], y=halo[:, 2], mode='markers', marker=dict(size=2, color='violet', opacity=0.8), name='halo'))
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
            title='DM wake sel YZ',
            xaxis_title='y [kpc]',
            yaxis_title='z [kpc]'
        )
        vel_yz.show()
    elif proj=='xz':
        x_i = lmc[0, 0]
        z_i = lmc[0, 2]
        vel_xz = ff.create_quiver(halo1[:,0], halo1[:,2], halo1[:,3], halo1[:,5], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        vel_xz.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xz.add_trace(go.Scatter(x=[x_i], y=[z_i], mode='markers', marker=dict(size=10, color='green', opacity=0.8), name='lmc_(0,0)'))
        vel_xz.add_trace(go.Scatter(x=halo[:, 0], y=halo[:, 2], mode='markers', marker=dict(size=2, color='violet', opacity=0.8), name='halo'))
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
            title='DM wake sel XZ',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        vel_xz.show()
    elif proj=='xy':
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy = ff.create_quiver(halo1[:,0], halo1[:,1], halo1[:,3], halo1[:,4], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        # vel_xy.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 1], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xy.add_trace(go.Scatter(x=[x_i], y=[y_i], mode='markers', marker=dict(size=10, color='green', opacity=0.8), name='lmc_(0,0)'))
        vel_xy.add_trace(go.Scatter(x=halo[:, 0], y=halo[:, 1], mode='markers', marker=dict(size=2, color='violet', opacity=0.8), name='halo'))
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
            title='DM wake sel XY',
            xaxis_title='x [kpc]',
            yaxis_title='y [kpc]'
        )
        vel_xy.show()



def trajectory_LMC(lmc, halo, slice_size, proj):
    '''
    :param halo:
    :param lmc:
    :param proj:
    :param slice_size:
    :return: vector field of the halo and lmc in the given projection
    '''
    if slice_size > halo.shape[0]:
        print('slice size is larger than the halo size, setting slice size to halo size')
        slice_size = halo.shape[0]
    random_slice = np.random.choice(halo.shape[0], slice_size, replace=False)
    halo1 = halo[random_slice]
    if proj=='yz':
        y_i = lmc[0, 1]
        z_i = lmc[0, 2]
        y_f = lmc[-1, 1]
        z_f = lmc[-1, 2]
        vel_yz = go.Figure()
        # vel_yz = ff.create_quiver(halo1[:,1], halo1[:,2], halo1[:,4], halo1[:,5], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        # vel_yz.add_trace(go.Scatter(x=lmc[:, 1], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_yz.add_trace(go.Scatter(x=[y_i], y=[z_i], mode='markers', marker=dict(size=10, color='orange', opacity=0.8), name='lmc_(0,0)'))
        vel_yz.add_trace(go.Scatter(x=[y_f], y=[z_f], mode='markers', marker=dict(size=10, color='purple', opacity=0.8), name='lmc_(0,0)'))
        vel_yz.add_trace(go.Scatter(x=halo[:, 1], y=halo[:, 2], mode='markers', marker=dict(size=2, color='blue', opacity=0.8), name='halo'))
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
            title='orbita LMC en YZ',
            xaxis_title='y [kpc]',
            yaxis_title='z [kpc]'
        )
        vel_yz.show()
    elif proj=='xz':
        x_i = lmc[0, 0]
        z_i = lmc[0, 2]
        x_f = lmc[-1, 0]
        z_f = lmc[-1, 2]
        # vel_xz = ff.create_quiver(halo1[:,0], halo1[:,2])
        vel_xz = go.Figure()
        # vel_xz.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xz.add_trace(go.Scatter(x=[x_i], y=[z_i], mode='markers', marker=dict(size=10, color='orange', opacity=0.8), name='lmc_(0,0)'))
        vel_xz.add_trace(go.Scatter(x=[x_f], y=[z_f], mode='markers', marker=dict(size=10, color='purple', opacity=0.8), name='lmc_(0,0)'))
        vel_xz.add_trace(go.Scatter(x=halo[:, 0], y=halo[:, 2], mode='markers', marker=dict(size=2, color='blue', opacity=0.8), name='halo'))
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
            title='Orbita LMC en XZ',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        vel_xz.show()
    elif proj=='xy':
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy = ff.create_quiver(halo1[:,0], halo1[:,1], halo1[:,3], halo1[:,4], arrow_scale=.25, name='vel', marker=dict(color='orange', size=5), line=dict(width=1),)
        # vel_xy.add_trace(go.Scatter(x=lmc[:, 0], y=lmc[:, 1], mode='lines', marker=dict(size=2, color='blue', opacity=0.8), name='lmc'))
        vel_xy.add_trace(go.Scatter(x=[x_i], y=[y_i], mode='markers', marker=dict(size=10, color='orange', opacity=0.8), name='lmc_(0,0)'))
        vel_xy.add_trace(go.Scatter(x=halo[:, 0], y=halo[:, 1], mode='markers', marker=dict(size=2, color='blue', opacity=0.8), name='halo'))
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
            title='Orbita LMC en XY',
            xaxis_title='x [kpc]',
            yaxis_title='y [kpc]'
        )
        vel_xy.show()


def vec_field2(pert, no_pert, lmc, proj):
    if proj == 'yz':
        vel_yz = ff.create_quiver(no_pert[:, 1], no_pert[:, 2],
                                  (pert[:, 1] - no_pert[:, 1]),
                                  (pert[:, 2] - no_pert[:, 2]),
                                  arrow_scale=.1,
                                  name='no pert', line=dict(width=0.5), marker=dict(size=2))
        x_i = lmc[0, 1]
        y_i = lmc[0, 2]
        vel_yz.add_trace(
            go.Scatter(x=pert[:, 1], y=pert[:, 2], marker=dict(size=1, color='red', opacity=0.8),
                       mode='markers', name='pert'))
        vel_yz.add_trace(
            go.Scatter(x=lmc[:, 1], y=lmc[:, 2], mode='lines', marker=dict(size=1, color='green', opacity=0.8),
                       name='lmc xy'))
        vel_yz.update_layout(title='vel wake  YZ', xaxis_title='x [kpc]', yaxis_title='y [kpc]', autosize=False, width=800,
                             height=800)
        vel_yz.add_trace(go.Scatter(x=[x_i], y=[y_i], mode='markers', marker=dict(size=10, color='blue', opacity=0.8),
                                    name='lmc_i'))
        vel_yz.show()
    elif proj == 'xz':
        vel_xz = ff.create_quiver(no_pert[:, 0], no_pert[:, 2],
                                    (pert[:, 0] - no_pert[:, 0]),
                                    (pert[:, 2] - no_pert[:, 2]),
                                    arrow_scale=.1,
                                    name='no pert', line=dict(width=0.5), marker=dict(size=2))
        x_i = lmc[0, 0]
        y_i = lmc[0, 2]
        vel_xz.add_trace(
            go.Scatter(x=pert[:, 0], y=pert[:, 2], marker=dict(size=1, color='red', opacity=0.8),
                          mode='markers', name='pert'))
        vel_xz.add_trace(
            go.Scatter(x=lmc[:, 0], y=lmc[:, 2], mode='lines', marker=dict(size=1, color='green', opacity=0.8),
                            name='lmc xy'))
        vel_xz.update_layout(title='vel wake  XZ', xaxis_title='x [kpc]', yaxis_title='y [kpc]', autosize=False, width=800,
                                 height=800)
        vel_xz.add_trace(go.Scatter(x=[x_i], y=[y_i], mode='markers', marker=dict(size=10, color='blue', opacity=0.8),
                                        name='lmc_i'))
        vel_xz.show()
    elif proj == 'xy':
        vel_xy = ff.create_quiver(no_pert[:, 0], no_pert[:, 1],
                                  (pert[:, 0] - no_pert[:, 0]),
                                  (pert[:, 1] - no_pert[:, 1]),
                                  arrow_scale=.1,
                                  name='no pert', line=dict(width=0.5), marker=dict(size=2))
        x_i = lmc[0, 0]
        y_i = lmc[0, 1]
        vel_xy.add_trace(
            go.Scatter(x=pert[:, 0], y=pert[:, 1], marker=dict(size=1, color='red', opacity=0.8),
                       mode='markers', name='pert'))
        vel_xy.add_trace(
            go.Scatter(x=lmc[:, 0], y=lmc[:, 1], mode='lines', marker=dict(size=1, color='green', opacity=0.8),
                       name='lmc xy'))
        vel_xy.update_layout(title='vel wake  XY', xaxis_title='x [kpc]', yaxis_title='y [kpc]', autosize=False, width=800,
                             height=800)
        vel_xy.add_trace(go.Scatter(x=[x_i], y=[y_i], mode='markers', marker=dict(size=10, color='blue', opacity=0.8),
                                    name='lmc_i'))
        vel_xy.show()

def E_L(halo, slice, proj):
    E_L = go.Figure()
    size = slice
    random_slice = np.random.choice(halo.shape[0], size, replace=False)
    if proj=='x':
        E_L.add_trace(go.Scatter(x=halo[:,11][random_slice], y=halo[:,15][random_slice],
                            mode='markers',
                            name='MW halo', marker=dict(size=1, color=halo[:,10][random_slice], colorscale='Viridis', showscale=True), hovertext=halo[:,7][random_slice]))
        E_L.update_layout(title='Energia vs momento angular X',
                            xaxis_title='Energia',
                            yaxis_title='Momento angular',
                            xaxis=dict(
                            title='Momento angular L_x (kpc km/s)',
                            scaleanchor="x",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            yaxis=dict(
                            title='Energia (km^2/s^2)',
                            scaleanchor="y",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            width=800,
                            height=800,
                            )
        E_L.show()
    elif proj=='y':
        E_L.add_trace(go.Scatter(x=halo[:,12][random_slice], y=halo[:,15][random_slice],
                            mode='markers',
                            name='MW halo', marker=dict(size=1, color=halo[:,10][random_slice], colorscale='Viridis', showscale=True), hovertext=halo[:,7][random_slice]))
        E_L.update_layout(title='Energia vs momento angular Y',
                            xaxis_title='Energia',
                            yaxis_title='Momento angular',
                            xaxis=dict(
                            title='Momento angular Ly (kpc km/s)',
                            scaleanchor="x",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            yaxis=dict(
                            title='Energia (km^2/s^2)',
                            scaleanchor="y",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            width=800,
                            height=800,
                            )
        E_L.show()
    elif proj=='z':
        E_L.add_trace(go.Scatter(x=halo[:,13][random_slice], y=halo[:,15][random_slice],
                            mode='markers',
                            name='MW halo', marker=dict(size=1, color=halo[:,10][random_slice], colorscale='Viridis', showscale=True), hovertext=halo[:,7][random_slice]))
        E_L.update_layout(title='Energia vs momento angular Z',
                            xaxis_title='Energia',
                            yaxis_title='Momento angular',
                            xaxis=dict(
                            title='Momento angular Lz (kpc km/s)',
                            scaleanchor="x",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            yaxis=dict(
                            title='Energia (km^2/s^2)',
                            scaleanchor="y",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            width=800,
                            height=800,
                            )
        E_L.show()
    elif proj=='mag':
        E_L.add_trace(go.Scatter(x=halo[:,10][random_slice], y=halo[:,15][random_slice],
                            mode='markers',
                            name='MW halo', marker=dict(size=1, color=halo[:,10][random_slice], colorscale='Viridis', showscale=True), hovertext=halo[:,7][random_slice]))
        E_L.update_layout(title='Energia vs Magnitud momento angular',
                            xaxis_title='Energia',
                            yaxis_title='L_mag Momento angular',
                            xaxis=dict(
                            title='Momento angular L (kpc km/s)',
                            scaleanchor="x",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            yaxis=dict(
                            title='Energia (km^2/s^2)',
                            scaleanchor="y",
                            scaleratio=1,
                            constrain='domain',
                            ),
                            width=800,
                            height=800,
                            )
        E_L.show()

def sel3(sel_3,rel_lmc, proj):
    if proj=='xy':
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(go.Scatter(x=sel_3[:, 0], y=sel_3[:, 1], mode='markers',
                                      marker=dict(size=3, color=sel_3[:, 10], colorscale='viridis', opacity=0.8,
                                                  colorbar=dict(
                                                      title='L [kpc km/s]',
                                                      titleside='top',
                                                      tickmode='array',
                                                      tickvals=[10000, 20000, 30000],
                                                      ticktext=['10000', '20000', '30000'],
                                                      ticks='outside',
                                                      thickness=20,
                                                      len=0.5,
                                                      y=0.5,
                                                      yanchor='middle',
                                                      x=1.1,
                                                      xanchor='left'
                                                  ), ), name='wake xy'))
        fig_sel3.add_trace(
            go.Scatter(x=rel_lmc[:, 0], y=rel_lmc[:, 1], mode='lines', marker=dict(size=2, color='blue', opacity=0.8, ),
                       name='LMC xy'))
        fig_sel3.add_trace(go.Scatter(x=rel_lmc[-1:, 0], y=rel_lmc[-1:, 1], mode='markers',
                                      marker=dict(size=10, color='purple', opacity=0.8, ), name='(x_f,y_f)'))
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
            title='DM wake sel XY',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        fig_sel3.show()
    elif proj=='xz':
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(go.Scatter(x=sel_3[:, 1], y=sel_3[:, 2], mode='markers',
                                      marker=dict(size=3, color=sel_3[:, 10], colorscale='viridis', opacity=0.8,
                                                  colorbar=dict(
                                                      title='L [kpc km/s]',
                                                      titleside='top',
                                                      tickmode='array',
                                                      tickvals=[10000, 20000, 30000],
                                                      ticktext=['10000', '20000', '30000'],
                                                      ticks='outside',
                                                      thickness=20,
                                                      len=0.5,
                                                      y=0.5,
                                                      yanchor='middle',
                                                      x=1.1,
                                                      xanchor='left'
                                                  ), ), name='wake yz'))
        fig_sel3.add_trace(
            go.Scatter(x=rel_lmc[:, 1], y=rel_lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8, ),
                       name='lmc yz'))
        fig_sel3.add_trace(go.Scatter(x=rel_lmc[-1:, 1], y=rel_lmc[-1:, 2], mode='markers',
                                      marker=dict(size=10, color='purple', opacity=0.8, ), name='(y_f,z_f)'))
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
            title='DM wake sel YZ',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        fig_sel3.show()
    elif proj=='yz':
        fig_sel3 = go.Figure()
        fig_sel3.add_trace(go.Scatter(x=sel_3[:, 0], y=sel_3[:, 2], mode='markers',
                                      marker=dict(size=3, color=sel_3[:, 10], colorscale='viridis', opacity=0.8,
                                                  colorbar=dict(
                                                      title='L [kpc km/s]',
                                                      titleside='top',
                                                      tickmode='array',
                                                      tickvals=[10000, 20000, 30000],
                                                      ticktext=['10000', '20000', '30000'],
                                                      ticks='outside',
                                                      thickness=20,
                                                      len=0.5,
                                                      y=0.5,
                                                      yanchor='middle',
                                                      x=1.1,
                                                      xanchor='left'
                                                  ), ), name='wake xz'))
        fig_sel3.add_trace(
            go.Scatter(x=rel_lmc[:, 0], y=rel_lmc[:, 2], mode='lines', marker=dict(size=2, color='blue', opacity=0.8, ),
                       name='lmc xz'))
        fig_sel3.add_trace(go.Scatter(x=rel_lmc[-1:, 0], y=rel_lmc[-1:, 2], mode='markers',
                                      marker=dict(size=10, color='purple', opacity=0.8, ), name='(x_f,z_f)'))
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
            title='DM wake sel XZ',
            xaxis_title='x [kpc]',
            yaxis_title='z [kpc]'
        )
        fig_sel3.show()




