import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R

def mechanical(data):
    """ Extract mechanical quantities from the data.
    Parameters
    ----------
    data : array
        Data read from the text file.
        Returns
        -------
        mechanical_data : array
            Mechanical quantities extracted from the data."""
    pos = data[:, 0:3]
    vel = data[:, 3:6]
    pot = data[:, 6]
    idn = data[:, 7]
    return pos, vel, pot, idn

def orbital(data):
    """ Extract orbital quantities from the data.
    Parameters
    ----------
    data : array
        Data read from the text file.
        Returns
        -------
        orbital_data : array
            Orbital quantities extracted from the data."""
    pos_mw = data[:, 0:3]
    pos_mw_lmc = data[:, 3:6]
    return pos_mw, pos_mw_lmc

def mag(data):
    """ Calculate  magnitudes of physical quantities from the data.
    Parameters
    ----------
    data : array
        Data read from the text file.
        Returns
        -------
        magnitudes_data : array
            Magnitudes calculated from data array."""
    magnitude = np.linalg.norm(data, axis=1, keepdims=True)
    # print(f"shape magnitude: {magnitude.shape}")
    return magnitude

def e_tot(vel, pot):
    """ Calculate energy of the system from the data.
    Parameters
    ----------
    pot : potential energy array
    vel : velocity array
        Returns
        -------
        energy_data : array
            Total energy calculated from data array."""
    print(f"shape vel: {vel.shape}")
    print(f"shape pot: {pot.shape}")
    kinetic = 0.5 * vel**2
    print(f"shape kinetic: {kinetic.shape}")
    E_total = kinetic + pot
    print(f"shape E_total: {E_total.shape}")
    return kinetic, E_total

def get_relative_orbit(orbit):
    mw = np.copy(orbit[:, 0:3])
    lmc = np.copy(orbit[:, 6:9])
    print(f"lmc shape: {lmc.shape}")
    print(f"lmc max: {np.max(lmc)}")
    pos_orbit_x = lmc[:, 0] - mw[:, 0]
    pos_orbit_y = lmc[:, 1] - mw[:, 1]
    pos_orbit_z = lmc[:, 2] - mw[:, 2]
    pos_orbit_x = pos_orbit_x[:, np.newaxis]
    pos_orbit_y = pos_orbit_y[:, np.newaxis]
    pos_orbit_z = pos_orbit_z[:, np.newaxis]
    # rel_orb = np.array([pos_orbit_x, pos_orbit_y, pos_orbit_z]).T
    rel_orb = np.concatenate((pos_orbit_x, pos_orbit_y, pos_orbit_z), axis=1)
    return rel_orb

def get_orbit_v(orbit):
    mw = np.copy(orbit[:, 3:6])
    lmc = np.copy(orbit[:, 9:12])
    vel_orbit_x = lmc[:, 0] - mw[:, 0]
    vel_orbit_y = lmc[:, 1] - mw[:, 1]
    vel_orbit_z = lmc[:, 2] - mw[:, 2]
    vel_orbit_x = vel_orbit_x[:, np.newaxis]
    vel_orbit_y = vel_orbit_y[:, np.newaxis]
    vel_orbit_z = vel_orbit_z[:, np.newaxis]
    # vel_orb = np.array([vel_orbit_x, vel_orbit_y, vel_orbit_z]).T
    vel_orb = np.concatenate((vel_orbit_x, vel_orbit_y, vel_orbit_z), axis=1)
    return vel_orb

def get_rot_orbit(orbit):
    pos_orb = get_relative_orbit(orbit)
    vel_orb = get_orbit_v(orbit)
    orb = np.concatenate((pos_orb, vel_orb), axis=1)
    print(f"shape orb: {orb.shape}")
    return orb
def get_ang_momentum(pos, vel):
    L = np.cross(pos, vel)
    # revisar esto
    Lx = np.array([L[:, 0]]).T
    Ly = np.array([L[:, 1]]).T
    Lz = np.array([L[:, 2]]).T
    L_mag = np.linalg.norm(L, axis=1, keepdims=True)
    # print(f"shape Lx: {Lx.shape}")
    # print(f"shape Ly: {Ly.shape}")
    # print(f"shape Lz: {Lz.shape}")
    # print(f"shape L_mag: {L_mag.shape}")
    total = len(L[:, 0])
    # L_total = np.sum(L[:, 0]) / total + np.sum(L[:, 1]) / total + np.sum(L[:, 2]) / total

    return L_mag, Lx, Ly, Lz

def get_orb_plane(pos):
    X = pos[:, 0:2]
    Y = pos[:, 2]
    reg = LinearRegression().fit(X, Y)

    a = reg.coef_[0]
    b = reg.coef_[1]
    c = -1
    coef = np.array([a, b, c])
    norm = np.linalg.norm(coef, axis=0)
    # print(norm)
    # # d = a * pos[:, 0] + b * pos[:, 1] + c * pos[:, 2]
    a /= norm
    b /= norm
    c /= norm
    # # # d /= np.linalg.norm(a, b, c)
    # # print(f"Plane equation: {a}x + {b}y + {c}z = 0")
    # print(f"The equation of the orbital plane is: {a:.3f}x + {b:.3f}y + {c:.3f}z = 0")
    normal_vector = np.array([a, b, c])
    print(f"Plane equation: {a: .3f}x + {b: .3f}y + {c: .3f}z = 0")
    print(f"Normal vector: {normal_vector}")
    # # #
    # z_angle = np.arccos(normal_vector[2]) * 180 / np.pi
    # y_angle = np.arccos(normal_vector[1]) * 180 / np.pi
    # x_angle = np.arccos(normal_vector[0]) * 180 / np.pi
    # print(f"Angle between normal vector and x-axis: {x_angle}",f"Angle between normal vector and y-axis: {y_angle}",f"Angle between normal vector and z-axis: {z_angle}")
    return normal_vector

def add_more_quant(mw,mw_lmc):
    #pos vel magnitudes
    pos_mw_mag = mag(mw[:,0:3])
    pos_mw_lmc_mag = mag(mw_lmc[:,0:3])
    vel_mw_mag =mag(mw[:,3:6])
    vel_mw_lmc_mag = mag(mw_lmc[:,3:6])
    # print("pos mag ********", pos_mw_mag.shape)
    # print("vel mag-------", vel_mw_mag.shape)
    #Energy
    #
    # k_mw, Et_mw = e_tot(vel_mw_mag[:,0], mw[:,6])
    # k_mw_lmc, Et_mw_lmc = e_tot(vel_mw_lmc_mag, mw_lmc[:,6])

    # L_mw_mag = np.ones((len(mw[:,0]),1))
    # print("shape L mag", L_mw_mag.shape)



    #Specific Angular momenta
    L_mw_mag,  Lx_mw, Ly_mw, Lz_mw = get_ang_momentum(mw[:,0:3], mw[:,3:6])
    L_mw_lmc_mag, Lx_mw_lmc, Ly_mw_lmc, Lz_mw_lmc = get_ang_momentum(mw_lmc[:,0:3], mw_lmc[:,3:6])

    # L_mw_mag = L_mw_mag[:, np.newaxis]
    # L_mw_lmc_mag, Lx_mw_lmc, Ly_mw_lmc, Lz_mw_lmc = get_ang_momentum(mw_lmc[:,0:3], mw_lmc[:,3:6])
    #Add to array
    # print("Antes", mw.shape)
    # print("shape L mag", L_mw_mag.shape)

    mw_arr = np.concatenate([mw[:,], pos_mw_mag, vel_mw_mag, L_mw_mag, Lx_mw, Ly_mw, Lz_mw], axis=1)
    lmc_arr = np.concatenate([mw_lmc[:,], pos_mw_lmc_mag, vel_mw_lmc_mag, L_mw_lmc_mag, Lx_mw, Ly_mw, Lz_mw], axis=1)


    #
    # mw_arr = np.concatenate([mw[:,], pos_mw_mag,vel_mw_mag, L_mw_mag, Lx_mw, Ly_mw, Lz_mw, k_mw, Et_mw], axis=1)
    # lmc_arr = np.concatenate([mw_lmc[:,], pos_mw_lmc_mag, vel_mw_lmc_mag, Lx_mw, Ly_mw, Lz_mw, k_mw_lmc, Et_mw_lmc], axis=1)

    # mw_arr = np.array((mw[:,0:3], pos_mw_mag, mw[:,3:6], vel_mw_mag, k_mw, mw[:,6],Et_mw, L_mw, L_mw_mag, L_tot_mw, Lx_mw, Ly_mw, Lz_mw, mw[:,7]))
    # mw_lmc_arr = np.array((mw_lmc[:,0:3], pos_mw_lmc_mag, mw_lmc[:,3:6], vel_mw_lmc_mag, k_mw_lmc, mw_lmc[:,6],Et_mw_lmc, L_mw_lmc, L_mw_lmc_mag, L_tot_mw_lmc, Lx_mw_lmc, Ly_mw_lmc, Lz_mw_lmc, mw_lmc[:,7]))

    return mw_arr, lmc_arr


