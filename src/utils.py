import matplotlib.pyplot as plt
from mat73 import loadmat
import indices
import sys
import numpy as np
from scipy.interpolate import interp2d
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle


sys.path.append('/home/david/Código/EAI-temporal-large-scale/src')


def get_indices(paths, fs, equalized=True):
    aci = []
    adi = []
    f1metric = []
    ndsi = []
    npeaks = []
    hf = []
    centroid = []
    spread = []

    for m in paths:
        pxx = loadmat(m)['Pxx']
        freq = np.linspace(0, fs//2, pxx.shape[0])
        pxx_db = 20*np.log10(pxx)

        if equalized:
            th = np.percentile(pxx_db, 50)
            mask = pxx_db < th
            pxx[mask] = 0
            pxx_db[mask] = pxx_db.min()

        ACIft_ = indices.ACIft(pxx_db)
        ADI_ = indices.ADIm(pxx, fs, wband=100)
        F1METRIC_ = indices.f1metric(pxx, freq, bio_band=(
            1200, 2000), tech_band=(200, 1000))
        NDSI_ = indices.NDSI(pxx, freq, bio_band=(
            1200, 2000), tech_band=(200, 1000))
        NP_ = indices.number_of_peaks(pxx_db, freq, slope=2, freq_dis=10)
        Hf_ = indices.spectral_entropy(pxx)
        CENTROID_, SPREAD_ = indices.spectral_centroid_spread(pxx, freq)

        aci.append(ACIft_)
        adi.append(ADI_)
        f1metric.append(F1METRIC_)
        ndsi.append(NDSI_)
        npeaks.append(NP_)
        hf.append(Hf_)
        centroid.append(CENTROID_)
        spread.append(SPREAD_)

    return aci, adi, f1metric, ndsi, npeaks, hf, centroid, spread


def interpolate_index(mat, xfactor, yfactor):
    x = np.linspace(0, mat.shape[1], mat.shape[1])
    y = np.linspace(0, mat.shape[0], mat.shape[0])

    f = interp2d(x, y, mat, kind='cubic')
    x2 = np.linspace(0, mat.shape[1], mat.shape[1]*yfactor)
    y2 = np.linspace(0, mat.shape[0], mat.shape[0]*xfactor)
    Z = f(x2, y2)
    return Z


class AcousticMap():
    def __init__(self, aci, adi, hm, ndsi, npeaks, hf, centroid, spread) -> None:
        self.aci = aci
        self.adi = adi
        self.hm = hm
        self.ndsi = ndsi
        self.npeaks = npeaks
        self.hf = hf
        self.centroid = centroid
        self.spread = spread

    def normalizeIndices(self):
        scaler = MinMaxScaler()
        aci_lt = np.array([x[:1425] for x in self.aci])
        aci_lt = scaler.fit_transform(aci_lt)

        scaler = MinMaxScaler()
        adi_lt = scaler.fit_transform(self.adi)

        scaler = MinMaxScaler()
        hm_lt = np.array([x[:1425] for x in self.hm])
        hm_lt = scaler.fit_transform(hm_lt)

        scaler = MinMaxScaler()
        ndsi_lt = np.array([x[:1425] for x in self.ndsi])
        ndsi_lt = scaler.fit_transform(ndsi_lt)

        scaler = MinMaxScaler()
        np_lt = np.array([x[:1425] for x in self.npeaks])
        np_lt = scaler.fit_transform(np_lt)

        scaler = MinMaxScaler()
        hf_lt = np.array([x[:1425] for x in self.hf])
        hf_lt = 1 - scaler.fit_transform(hf_lt)

        scaler = MinMaxScaler()
        centroid_lt = np.array([x[:1425] for x in self.centroid])
        centroid_lt = 1 - scaler.fit_transform(centroid_lt)

        scaler = MinMaxScaler()
        spread_lt = np.array([x[:1425] for x in self.spread])
        spread_lt = scaler.fit_transform(spread_lt)

        self.aci = aci_lt
        self.adi = adi_lt
        self.hm = hm_lt
        self.ndsi = ndsi_lt
        self.npeaks = np_lt
        self.hf = hf_lt
        self.centroid = centroid_lt
        self.spread = spread_lt

        return aci_lt, adi_lt, hm_lt, ndsi_lt, np_lt, hf_lt, centroid_lt, spread_lt

    def __plotIndex(self, index, name, ax, normalized=True):
        ind_lt = np.array([x[:1425] for x in index])
        ind_int = interpolate_index(ind_lt, 50, 1)
        if normalized:
            ax.pcolormesh(ind_int.T, cmap='jet', vmin=0, vmax=1)
        else:
            ax.pcolormesh(ind_int.T, cmap='jet', vmin=np.percentile(
                ind_int, 1), vmax=np.percentile(ind_int, 99))

        ax.set_title(name)
        ax.set_xlabel('Days')
        ax.set_ylabel('Hours')
        ax.xaxis.set_major_locator(
            FixedLocator(np.linspace(0, 29*50, 4, dtype=int)))
        ax.set_xticklabels(np.linspace(1, 29, 4, dtype=int))
        ax.yaxis.set_major_locator(
            FixedLocator(np.linspace(0, 1425, 4, dtype=int)))
        ax.set_yticklabels(np.linspace(0, 24, 4, dtype=int))

    def showMap(self, normalized=True):
        if normalized:
            self.normalizeIndices()

        fig, ax = plt.subplots(3, 3, figsize=(10, 9))

        # ACI
        self.__plotIndex(self.aci, 'ACI', ax[0, 0], True)

        # ax[0,0].set_yticklabels(np.linspace(0,24,7, dtype=int))

        # CENTROID
        self.__plotIndex(self.centroid, 'CENTROID', ax[0, 1], True)

        # HM
        self.__plotIndex(self.hm, 'HM', ax[0, 2], True)

        # NDSI
        self.__plotIndex(self.ndsi, 'NDSI', ax[1, 0], True)

        # NP
        self.__plotIndex(self.npeaks, 'NP', ax[1, 1], True)

        # HF
        self.__plotIndex(self.hf, 'HF', ax[1, 2], True)

        # SPREAD
        self.__plotIndex(self.spread, 'SPREAD', ax[2, 0], True)

        fig.tight_layout()

        plt.show()
        return fig, ax


def plotrect(ax, px1, px2, py1, py2, c, day):
    if px1 > px2:
        width = 1440 - px1 + px2
    else:
        width = px2 - px1

    pxw = px1 + day*1440
    pyh = py2 - py1
    ax.add_patch(Rectangle((pxw, py1),
                           width,
                           pyh,
                           edgecolor=c, facecolor='none', lw=1))


def apply_mask(mat, x1, x2, y1, y2):
    try:
        if x1 > x2:
            width = 1440 - x1
        else:
            width = x2 - x1
        for i in range(int(x1), int(width+x1)):
            mat[int(y1):int(y2), i] = 1
    except:
        pass

    return mat
