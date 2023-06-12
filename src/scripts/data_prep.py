import load as ld
import rotations as rot
import phys_quant as pq
import numpy as np

mw = ld.txt('rand_mwb1_000')
mw_lmc = ld.txt('rand_mwlmcb1_110')
orbit = ld.orbit('LMC5_100Mb1_orbit')

#Get physical quantities
pos_mw, vel_mw, pot_mw, id_mw = pq.mechanical(mw)
pos_mw_lmc, vel_mw_lmc, pot_mw_lmc, id_mw_lmc = pq.mechanical(mw_lmc)
pos_mw_mag = pq.mag(pos_mw)
pos_mw_lmc_mag = pq.mag(pos_mw_lmc)
vel_mw_mag = pq.mag(vel_mw)
vel_mw_lmc_mag = pq.mag(vel_mw_lmc)
# print(pq.e_tot(vel_mw_mag, pot_mw))
k_mw, Et_mw, = pq.e_tot(vel_mw_mag, pot_mw[:, np.newaxis])
k_mw_lmc, Et_mw_lmc, = pq.e_tot(vel_mw_lmc_mag, pot_mw_lmc[:, np.newaxis])



#Get relative orbit of LMC
rel_lmc_orbit = pq.get_relative_orbit(orbit)

n_vector= pq.get_orb_plane(rel_lmc_orbit)
spherical_n = rot.cartesian_to_spherical(n_vector)
rot_n = rot.rotate(spherical_n, n_vector)

tolerable_error = 1e-15
#This function checks if the rotation aligns the orbital plane with the z-axis
def check_rotation(normal_vector,pot_mw, id_mw, pot_mw_lmc, id_mw_lmc):
    if abs(normal_vector[2] -1) < tolerable_error:
        print("The rotation aligns orbital plane with the z-axis.")
        pos_mw_rot = rot.rotate(spherical_n, pos_mw)
        pos_mw_lmc_rot = rot.rotate(spherical_n, pos_mw_lmc)
        vel_mw_rot = rot.rotate(spherical_n, vel_mw)
        vel_mw_lmc_rot = rot.rotate(spherical_n, vel_mw_lmc)
        pot_mw_rs = pot_mw[:, np.newaxis]
        id_mw_rs = id_mw[:, np.newaxis]
        pot_mw_lmc_rs = pot_mw_lmc[:, np.newaxis]
        id_mw_lmc_rs = id_mw_lmc[:, np.newaxis]
        mw_rot = np.concatenate((pos_mw_rot, vel_mw_rot, pot_mw_rs, id_mw_rs), axis=1)
        mw_lmc_rot = np.concatenate((pos_mw_lmc_rot, vel_mw_lmc_rot, pot_mw_lmc_rs, id_mw_lmc_rs), axis=1)
        mw_L, mw_lmc_L = pq.add_more_quant(mw_rot, mw_lmc_rot)
        mw_total = np.concatenate((mw_L, k_mw, Et_mw ), axis=1)
        mw_lmc_total = np.concatenate((mw_lmc_L, k_mw_lmc, Et_mw_lmc), axis=1)
        orbit_mw_rot_pos = rot.rotate(spherical_n, orbit[:, 0:3])
        orbit_mw_rot_vel = rot.rotate(spherical_n, orbit[:, 3:6])
        orbit_lmc_rot_pos = rot.rotate(spherical_n, orbit[:, 6:9])
        orbit_lmc_rot_vel = rot.rotate(spherical_n, orbit[:, 9:12])
        lmc_rot = np.concatenate((orbit_mw_rot_pos, orbit_mw_rot_vel, orbit_lmc_rot_pos, orbit_lmc_rot_vel, ), axis=1)
        return mw_total,mw_lmc_total, lmc_rot
    else:
        print("The rotation does not align orbital plane with the z-axis.")

# mw_rot, mw_lmc_rot, lmc_rot = check_rotation(rot_n)
data = check_rotation(rot_n, pot_mw, id_mw, pot_mw_lmc, id_mw_lmc)
if data is not None:
    mw_rot, mw_lmc_rot, lmc_rot = data
    print(mw_rot.shape, mw_lmc_rot.shape, lmc_rot.shape)
    np.savetxt('../../data/mw_rot.txt', mw_rot)
    np.savetxt('../../data/mw_lmc_rot.txt', mw_lmc_rot)
    np.savetxt('../../data/lmc_rot.txt', lmc_rot)
else:
    print("data is None")

