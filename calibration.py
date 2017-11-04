"""
Calibration of the alpao mirror

for random values strategy ONLY

mweigert@mpi-cbg.de


The modes that are ill captured by the mirror are (n,\pm 1) with n odd,
e.g. (3,1), coma, (5,1), (7,1)...


"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from zernike import zernike


def pinv_reg(M, reg=0):
    return np.dot(np.linalg.inv(np.dot(M.T, M) + reg * np.identity(M.shape[1])), M.T)


class Calibration(object):
    save_attr = ["M", "M_inv", "cal_vals", "cal_zerns", "flat_zerns", "flat_vals"]
    # omit piston
    zmodes_alpao_order = [(n, m) for n in range(13) for m in np.arange(-n, n + 1, 2)]
    n_alpao_modes = 66
    n_acc = 97

    def __init__(self, fname=""):
        if fname:
            self.load(fname)

        self.acc_coords = self.actuator_coords()

        _x = np.linspace(-1, 1, 100)
        Y, X = np.meshgrid(_x, _x, indexing="ij")
        rho = np.hypot(X, Y)
        phi = np.arctan2(Y, X)

        self.zern_modes = np.stack([zernike(n, m, rho, phi) for n, m in self.zmodes_alpao_order])
        for z in self.zern_modes:
            z[rho > 1] = np.nan

        Y, X = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), indexing="ij")
        self.accs_mask = np.bitwise_and(np.bitwise_and(-X - 1 + Y - 1 < 6, X - 1 + Y - 1 < 6),
                                        np.bitwise_and(X - 1 - Y - 1 < 6, -X - 1 - Y - 1 < 6))

    def actuator_coords(self):
        """returns the coordinates of the actuators

        return rs.shape = (2,97)
        with rs[:,0], rs[:,1] are the y and x coordinates

        in integer coordinates [y_0, x_0] from -5..5
        """

        mask = np.ones((11, 11), np.bool)
        for i in range(0, 3):
            for j in range(3 - i):
                mask[i, j] = False
        mask = np.bitwise_and(mask, mask[::-1])
        mask = np.bitwise_and(mask, mask[:, ::-1])
        rs = np.stack(np.where(mask)).T - 5
        return rs

    def load_calibration(self, fname, remove_average_modes=1):
        """
        loads data from a calibration file and builds the influence matrix
        """
        d = np.genfromtxt(fname)

        n_line = d.shape[-1]
        if n_line == self.n_acc + self.n_alpao_modes:
            print("random acc, alpao modes")
            self._load_calibration_mode_random_acc_alpao_modes(d, remove_average_modes=remove_average_modes)
        elif n_line == 4 + self.n_alpao_modes:
            print("single acc, alpao modes")
            self._load_calibration_mode_single_acc_alpao_modes(d, remove_average_modes=remove_average_modes)

        else:
            raise ValueError("wrong shape of input data")

    def _load_calibration_mode_random_acc_alpao_modes(self, d, remove_average_modes):

        assert d.shape[-1] == self.n_acc + self.n_alpao_modes

        n_vals = d.shape[0]

        self.cal_vals = d[:, :self.n_acc]
        self.cal_zerns = d[:, self.n_acc:]

        print("removing first %s averages" % remove_average_modes)

        self.cal_zerns[:, :remove_average_modes] -= np.mean(self.cal_zerns[:, :remove_average_modes], axis=0)

        model = linear_model.LinearRegression()
        model.fit(self.cal_vals, self.cal_zerns)

        self.M = model.coef_
        self.flat_zerns = model.intercept_

    def _load_calibration_mode_single_acc_alpao_modes(self, d, remove_average_modes):
        assert d.shape[-1] == 4 + self.n_alpao_modes

        n_vals = d.shape[0] // self.n_acc
        assert n_vals % 2 == 1

        self.cal_vals = d[:, 3].reshape((self.n_acc, n_vals))

        # omit piston
        # self.cal_zerns = d[:, 4:].reshape((self.n_acc, n_vals, -1))

        self.cal_zerns = d[:, 4:].reshape((self.n_acc, n_vals, -1))
        self.cal_zerns[:, : remove_average_modes] -= np.mean(self.cal_zerns[:remove_average_modes], axis=1)

        self.flat_zerns = np.mean(self.cal_zerns[:, n_vals // 2, :], 0)

        # fit the influence matrix
        model = linear_model.RANSACRegressor(linear_model.LinearRegression())

        # create the influence matrix

        M = np.zeros((self.n_alpao_modes, self.n_acc))

        cal_zerns_normed = self.cal_zerns - self.flat_zerns

        for acc in range(self.n_acc):
            print("fitting actuator # %s" % acc)
            for mode in range(self.n_alpao_modes):
                model.fit(self.cal_vals[acc, :, np.newaxis], self.cal_zerns[acc, :, mode])
                M[mode, acc] = model.estimator_.coef_

        self.M = M

    def _construct_zmodes_from_mask(self, wfs_raw):

        # get the pupil shape and center from the data

        # crop
        inds = np.where(np.isnan(wfs_raw[0]))
        wfs = wfs_raw[:, np.amin(inds[0]):np.amax(inds[0]), np.amin(inds[1]):np.amax(inds[1])]

        # find center/pupil

        pupil = np.bitwise_not(np.isnan(wfs[0]))
        pupil_sum = np.sum(pupil)
        _Y, _X = np.meshgrid(np.arange(wfs[0].shape[0]), np.arange(wfs[0].shape[1]), indexing="ij")
        y0, x0 = 1. * np.sum(_Y * pupil) / pupil_sum, 1. * np.sum(_X * pupil) / pupil_sum
        rad = np.sqrt(pupil_sum / np.pi)
        _R = np.sqrt((_X - x0) ** 2 + (_Y - y0) ** 2) / rad
        _P = np.arctan2(_Y - y0, _X - x0)

        self._wf_dx = 1. / rad
        self.zern_modes_wf = np.stack(
            [1. / np.sqrt(np.pi) * zernike(n, m, _R, _P) for n, m in self.zmodes_alpao_order[:self.n_custom_modes]])
        self._wfs = wfs

    def fit(self, reg=1.e-10, n_ignore=1):

        self.M_inv = pinv_reg(self.M[n_ignore:], reg)
        if n_ignore > 0:
            self.M_inv = np.concatenate([np.zeros((97, n_ignore)), self.M_inv], axis=1)

        self.flat_vals = -np.dot(self.M_inv, self.flat_zerns)

    def save(self, fname):
        save_dict = dict([(attr, getattr(self, attr)) for attr in self.save_attr])
        np.savez(fname, **save_dict)

    def load(self, fname):
        f = np.load(fname)
        self.__dict__.update(dict(f.items()))
        if hasattr(self, "cal_zerns_mean"):
            self.flat_zerns = self.cal_zerns_mean

        self.n_vals = f["cal_vals"].shape[1]
        self.flat_vals = -np.dot(self.M_inv, self.flat_zerns)

    def wavefront(self, zerns):
        if np.isscalar(zerns):
            _z = np.zeros(len(self.zern_modes))
            _z[zerns] = 1.
            zerns = _z

        return np.sum([self.zern_modes[i] * z for i, z in enumerate(zerns)], axis=0)

    def acc_to_zern(self, vals):
        return self.flat_zerns + np.dot(self.M, vals)

    def zern_to_acc(self, zerns, zval=1.):
        """zerns can be an array of length n_modes or a integer number in which case
         that single mode is picked
        """
        if np.isscalar(zerns):
            _z = np.zeros(self.M_inv.shape[-1])
            _z[zerns] = zval
            zerns = _z

        return self.flat_vals + np.dot(self.M_inv, zerns)

    def plot_val_series(self, acc_id, remove_flat=True):
        plt.clf()
        for i in range(self.n_vals):
            plt.subplot(5, 5, i + 1)
            val = self.cal_zerns[acc_id, i, :]
            if remove_flat:
                val -= self.flat_zerns

            plt.imshow(self.wavefront(val),
                       vmin=-0.3, vmax=.3, cmap="viridis")
            plt.title(self.cal_vals[acc_id, i], fontsize=8)
            plt.axis("off")
        plt.suptitle("actuator id #%s" % acc_id)

    def plot_acc(self, vals=None, with_colorbar=True, **kwargs):
        if vals is None:
            vals = self.flat_vals

        ind = np.where(self.accs_mask.flatten())
        res = np.zeros(self.accs_mask.flatten().shape)
        res[ind] = vals
        res = res.reshape(self.accs_mask.shape)
        res[np.bitwise_not(self.accs_mask)] = np.nan
        plt.cla()
        plt.imshow(res, **kwargs)
        if with_colorbar:
            plt.colorbar()
        plt.axis("off")

    def plot_influences(self, accs=None):
        if accs is None:
            accs = np.arange(97)

        n = len(accs)
        w = int(np.ceil(1.7 * np.sqrt(n)))
        h = int(np.ceil(n / w))

        plt.figure()
        plt.clf()
        for j in range(h):
            for i in range(w):
                k = i * h + j
                acc_vals = np.zeros(97)
                acc_vals[k] = 1.
                m = self.wavefront(self.acc_to_zern(acc_vals))

                plt.subplot(h, w, k + 1)
                plt.imshow(m, cmap="hot", vmin=-0, vmax=6)
                plt.axis("off")

    def save_flat_Minv_txt(self, fname_suffix=""):
        M_inv = self.M_inv.copy()
        if M_inv.shape[-1] < 66:
            M_inv = np.hstack([np.zeros((97, 66 - M_inv.shape[-1])), M_inv])
        elif M_inv.shape[-1] > 66:
            M_inv = np.hstack([np.zeros((97, 1)), M_inv])
            M_inv = M_inv[:, :66]

        np.savetxt("data/calib_Minv_%s.txt" % fname_suffix, M_inv, delimiter=",")
        np.savetxt("data/calib_flat_%s.txt" % fname_suffix, self.flat_vals[np.newaxis, :], delimiter=",")

    def reconstruct_zerns(self, zerns):
        a = self.zern_to_acc(zerns)
        return self.acc_to_zern(a)

    def plot_reco_mode(self, n_mode):
        def normalize(a):
            mi = np.nanmin(a)
            ma = np.nanmax(a)
            return -1. + 2. * (a - mi) / (ma - mi)

        def zernike_mask(rs, n, m):
            _R = 1. / 5 * np.sqrt(np.sum(rs ** 2, axis=1))
            _P = np.arctan2(rs[:, 0], rs[::-1, 1])
            return zernike(n, m, _R, _P)

        a1 = self.zern_to_acc(n_mode)
        rs = self.actuator_coords()
        a2 = zernike_mask(rs, *c.zmodes_alpao_order[n_mode])

        w1 = normalize(self.wavefront(self.acc_to_zern(a1)))
        w2 = normalize(self.wavefront(self.acc_to_zern(a2)))
        z0 = normalize(self.zern_modes[n_mode])
        print("with Minv\t", np.nanmean(np.abs(z0 - w1)))
        print("without \t", np.nanmean(np.abs(z0 - w2)))

        plt.figure(1)
        plt.clf()
        for i, (a, w) in enumerate(zip((a1, a2), (w1, w2))):
            plt.subplot(3, 2, i + 1)
            plt.imshow(w, vmin=-1, vmax=1)
            plt.axis("off")
            plt.title("w%s" % (i + 1))
            plt.subplot(3, 2, i + 1 + 2)
            plt.imshow(w - z0, vmin=-1, vmax=1)
            plt.axis("off")
            plt.title("$\Delta = %.5f$" % np.nanmax(np.abs(z0 - w)), fontsize=8)
            plt.subplot(3, 2, i + 1 + 4)
            c.plot_acc(a)
        plt.suptitle("z_mode = %s" % n_mode)
        plt.show()


def M_inv_from_mirror():
    from mirror import AlpaoMirror
    mir = AlpaoMirror()

    M = np.zeros((97, 66))
    for i, (n, m) in enumerate(Calibration.zmodes_alpao_order):
        target = np.zeros_like(mir.phase)
        target[mir.mask] = mir.zernike(n, m)[mir.mask]
        M[:, i] = mir.resp_to_acc(target)

    return M


if __name__ == '__main__':
    c = Calibration()
    c.load_calibration("data/AlpaoCalibration22_with_tip.txt", remove_average_modes=3)

    c.fit(1.e-4, n_ignore=1)

    # c.save_flat_Minv_txt("22")


    # # pure defocus
    # z_vals = tuple((3, dx) for dx in .2 * np.linspace(-1, 1, 7))
    # accs = np.stack(c.zern_to_acc(z, val) for z, val in z_vals)
    # np.savetxt("ao_accs/7_0.2_defocus.txt", accs, delimiter=",")
    # np.savetxt("ao_accs/zvals_7_0.2_defocus.txt", z_vals, delimiter=",", fmt = str("%.4f"))
    #
    # # defocus and astigmat
    # z_vals = ((0,0),)+tuple((z,pref*.1) for z in (3, 4, 5) for pref in (-1,1))
    # accs = np.stack(c.zern_to_acc(z, val) for z, val in z_vals)
    # np.savetxt("ao_accs/7_0.2.txt", accs, delimiter=",")
    # np.savetxt("ao_accs/zvals_7_0.2.txt", z_vals, delimiter=",", fmt = str("%.4f"))
    #
    # # defocus, astigmat, trefoil
    # z_vals = ((0,0),)+tuple((z,pref*.1) for z in (3, 4, 5,6,9) for pref in (-1,1))
    # accs = np.stack(c.zern_to_acc(z, val) for z, val in z_vals)
    # np.savetxt("ao_accs/11_0.2.txt", accs, delimiter=",")
    # np.savetxt("ao_accs/zvals_11_0.2.txt", z_vals, delimiter=",", fmt = str("%.4f"))
    #
    #
    # # all
    # z_vals = tuple((3, dx) for dx in .2 * np.linspace(-1, 1, 7))+tuple((z,pref*.1) for z in (3, 4, 5,6,9) for pref in (-1,1))
    # accs = np.stack(c.zern_to_acc(z, val) for z, val in z_vals)
    # np.savetxt("ao_accs/all.txt", accs, delimiter=",")
    # np.savetxt("ao_accs/zvals_all.txt", z_vals, delimiter=",", fmt = str("%.4f"))
