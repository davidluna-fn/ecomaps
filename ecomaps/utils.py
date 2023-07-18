import datetime
import sys

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler

from . import indices
from mat73 import loadmat


def getLongMatrix(paths, fs, nfft, temporal_bins, days):
    """Obtiene una matriz larga a partir de archivos de datos.

    Args:
        paths (List[str]): Lista de rutas de archivos.
        nfft (int): Tamaño de la matriz en la primera dimensión.

    Returns:
        np.ndarray: Matriz resultante de dimensiones (nfft, m), donde 'm' es la suma de todas las dimensiones de los archivos cargados.

    Raises:
        KeyError: Si la clave 'Pxx' no está presente en algún archivo.
        ValueError: Si la forma de 'pxx' no es válida para concatenación.

    """
    Sxx = np.zeros((nfft, 1))
    for s in range(len(paths)):
        pxx = loadmat(paths[s])['Pxx']
        if pxx.shape[1] < 1440:
            temp = pxx[:, -1].reshape(pxx.shape[0], 1)
            pxx = np.hstack([pxx, np.repeat(temp, 1440 - pxx.shape[1], 1)])

        Sxx = np.hstack([Sxx, pxx])

    Sxx = Sxx[:, 1:]

    base = datetime.datetime(2015, 2, 4)
    t = np.array([base + datetime.timedelta(minutes=1)*i
                 for i in range(temporal_bins*days)])
    f = np.linspace(0, fs//2, nfft)

    return Sxx, t, f


def get_indices(Sxx, fs, freq, temporal_bins, days, equalized=True):
    aci = []
    f1metric = []
    ndsi = []
    npeaks = []
    hf = []
    centroid = []
    spread = []
    b1 = []
    b2 = []

    Sxx = Sxx.reshape((Sxx.shape[0], days, temporal_bins))

    for m in range(Sxx.shape[1]):
        pxx = Sxx[:, m, :]
        pxx_db = 20*np.log10(Sxx[:, m, :])

        if equalized:
            th = np.percentile(pxx_db, 50)
            mask = pxx_db < th
            pxx[mask] = 0
            pxx_db[mask] = pxx_db.min()

        ACIft_ = indices.ACIft(pxx_db)
        F1METRIC_ = indices.f1metric(pxx, freq, bio_band=(
            1200, 2000), tech_band=(200, 1000))
        NDSI_ = indices.NDSI(pxx, freq, bio_band=(
            1200, 2000), tech_band=(200, 1000))
        NP_ = indices.number_of_peaks(pxx_db, freq, slope=2, freq_dis=10)
        Hf_ = indices.spectral_entropy(pxx)
        CENTROID_, SPREAD_ = indices.spectral_centroid_spread(pxx, freq)
        B1 = indices.beta(pxx, freq, (200, 1000))
        B2 = indices.beta(pxx, freq, (1200, 2000))

        aci.append(ACIft_)
        f1metric.append(F1METRIC_)
        ndsi.append(NDSI_)
        npeaks.append(NP_)
        hf.append(Hf_)
        centroid.append(CENTROID_)
        spread.append(SPREAD_)
        b1.append(B1)
        b2.append(B2)

    return aci, f1metric, ndsi, npeaks, hf, centroid, b1, b2


def interpolate_index(mat, xfactor, yfactor):
    x = np.linspace(0, mat.shape[1], mat.shape[1])
    y = np.linspace(0, mat.shape[0], mat.shape[0])

    f = interp2d(x, y, mat, kind='cubic')
    x2 = np.linspace(0, mat.shape[1], mat.shape[1]*yfactor)
    y2 = np.linspace(0, mat.shape[0], mat.shape[0]*xfactor)
    Z = f(x2, y2)
    return Z


class AcousticMap():
    def __init__(self, aci, hm, ndsi, npeaks, hf, centroid, b1, b2) -> None:
        self.aci = aci
        self.hm = hm
        self.ndsi = ndsi
        self.npeaks = npeaks
        self.hf = hf
        self.centroid = centroid
        self.b1 = b1
        self.b2 = b2

    def normalizeIndices(self):
        scaler = MinMaxScaler()
        aci_lt = np.array([x[:1425] for x in self.aci])
        aci_lt = scaler.fit_transform(aci_lt)

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

        scaler = MinMaxScaler()
        b1_lt = np.array([x[:1425] for x in self.b1])
        b1_lt = 1 - scaler.fit_transform(b1_lt)

        self.aci = aci_lt
        self.adi = adi_lt
        self.hm = hm_lt
        self.ndsi = ndsi_lt
        self.npeaks = np_lt
        self.hf = hf_lt
        self.centroid = centroid_lt
        self.spread = spread_lt
        self.b1 = b1_lt

        return aci_lt, hm_lt, ndsi_lt, np_lt, hf_lt, centroid_lt, spread_lt, b1_lt

    def __plotIndex(self, index, name, ax, sun_hour=None, normalized=True):
        ind_int = interpolate_index(np.array(index), 50, 1)
        x = np.linspace(1, 29, 29*50, endpoint=True)
        y = np.linspace(0, 24, 1440)

        if normalized:
            ax.pcolormesh(x, y, ind_int.T, cmap='jet', vmin=0, vmax=1)
        else:
            ax.pcolormesh(x, y, ind_int.T, cmap='jet', vmin=np.percentile(
                ind_int, 1), vmax=np.percentile(ind_int, 99.99))

        ax.set_title(name)
        ax.set_xlabel('Days')
        ax.set_ylabel('Hours')
        # ax.xaxis.set_major_locator(
        #    FixedLocator(np.linspace(0, 29*50, 4, dtype=int)))
        # ax.set_xticklabels(np.linspace(1, 29, 4, dtype=int))
        # ax.yaxis.set_major_locator(
        #    FixedLocator(np.linspace(0, 1425, 4, dtype=int)))
        # ax.set_yticklabels(np.linspace(0, 24, 4, dtype=int))
        if sun_hour.any() != None:
            sunrise = [x.hour + x.minute/60 for x in sun_hour[:, 0]]
            sunset = [x.hour + x.minute/60 for x in sun_hour[:, 1]]

            ax.plot(np.linspace(1, 29, 29, endpoint=True), sunrise,
                    color='white', linestyle='dashed', linewidth=0.8)
            ax.plot(np.linspace(1, 29, 29, endpoint=True), sunset,
                    color='white', linestyle='dashed', linewidth=0.8)

    def showMap(self, sun_hour=None, normalized=True):
        if normalized:
            self.normalizeIndices()

        fig, ax = plt.subplots(2, 4, figsize=(10, 5))

        # ACI
        self.__plotIndex(self.aci, 'ACI', ax[0, 0], sun_hour, normalized)

        # CENTROID
        self.__plotIndex(self.centroid, 'CENTROID',
                         ax[0, 1], sun_hour, normalized)

        # HM
        self.__plotIndex(self.hm, 'HM', ax[0, 2], sun_hour, normalized)

        # NDSI
        self.__plotIndex(self.ndsi, 'NDSI', ax[0, 3], sun_hour, normalized)

        # NP
        self.__plotIndex(self.npeaks, 'NP', ax[1, 0], sun_hour, normalized)

        # HF
        self.__plotIndex(self.hf, 'HF', ax[1, 1], sun_hour, normalized)

        # SPREAD
        self.__plotIndex(self.b1, 'B1', ax[1, 2], sun_hour, normalized)
        self.__plotIndex(self.b2, 'B2', ax[1, 3], sun_hour, normalized)

        fig.tight_layout()

        plt.show()
        return fig, ax


def plotrect(ax, px1, px2, py1, py2, c, day):
    if px1 > px2:
        width = datetime.timedelta(
            minutes=1440 - px1.hour*60 - px1.minute + px2.hour*60 + px2.minute)
    else:
        width = px2 - px1

    pyh = py2 - py1
    ax.add_patch(Rectangle((px1, py1),
                           width,
                           pyh,
                           edgecolor=c, facecolor='none', lw=1))


def apply_mask(mat, x1, x2, y1, y2, v):
    try:
        if x1 > x2:
            width = 1440 - x1
        else:
            width = x2 - x1
        for i in range(int(x1), int(width+x1)):
            mat[int(y1):int(y2), i] = v
    except:
        pass

    return mat


def plotLongSpectrogram(X, nsubplots, ndays, nbins, show=True):
    fig, ax = plt.subplots(nsubplots, 1, figsize=(12, 10),)
    if nsubplots == 1:
        ax = [ax]

    for p in range(nsubplots):
        ax[p].pcolormesh(
            X[p*nbins*ndays:(p+1)*nbins*ndays, :].T, cmap='jet')
        ax[p].set_xlabel('Days')
        ax[p].set_ylabel('Frequency [Hz]')
        ax[p].set_yticklabels(np.linspace(0, 11025//2, 6, dtype=int)//1000)
        ax[p].xaxis.set_major_locator(FixedLocator(
            np.linspace(0, ndays*nbins, ndays+1, dtype=int)[:-1]))
        ax[p].set_xticklabels(np.linspace(
            p*ndays, ndays*(p+1), ndays, dtype=int))
        ax[p].set_ylim(0, 400)
        ax[p].set_xlim(0, nbins*ndays)

    if show:
        fig.tight_layout()
        fig.show()
    return fig, ax


def plotLongSpectrogramChoirs(X, t, f, nsubplots, ndays, nbins, choirs_temp, choirs_freq, figsize=(12, 10), bias=1):
    fig, ax = plt.subplots(nsubplots, 1, figsize=figsize,)
    if nsubplots == 1:
        ax = [ax]

    event1 = choirs_temp['Ann_Ch2'][:, :-1]
    event2 = choirs_temp['Ann_Ch3'][:, :-1]
    event3 = choirs_temp['Ann_Ch4'][:, :-1]
    event4 = choirs_temp['Ann_Ch5'][:, :-1]
    flow = choirs_freq['BW_fLow'][:28, :].T
    fhigh = choirs_freq['BW_fHig'][:28, :].T

    count = 0 + bias - 1
    count2 = 0

    for p in range(nsubplots):
        t[p*nbins*ndays:(p+1)*nbins*ndays]
        ax[p].pcolormesh(t[p*nbins*ndays:(p+1)*nbins*ndays], f,
                         X[:, p*nbins*ndays:(p+1)*nbins*ndays], cmap='jet')
        ax[p].set_xlabel('Days')
        ax[p].set_ylabel('Frequency [Hz]')
        ax[p].set_xticks(
            t[np.linspace((p*nbins*ndays)+nbins, (p+1)*nbins*ndays, ndays, dtype=int)])

        for d in range(ndays):
            e1 = event1[:, count]
            e2 = event2[:, count]
            e3 = event3[:, count]
            e4 = event4[:, count]
            try:
                plotrect(ax[p], t[int(count2*nbins + e1[0])], t[int(count2*nbins + e1[2])],
                         flow[1, count], fhigh[1, count], 'black', d)
            except:
                pass

            try:
                plotrect(ax[p], t[int(count2*nbins + e2[0])], t[int(count2*nbins + e2[2])],
                         flow[2, count], fhigh[2, count], 'green', d)
            except:
                pass
            try:
                plotrect(ax[p], t[int(count2*nbins + e3[0])], t[int(count2*nbins + e3[2])],
                         flow[3, count], fhigh[3, count], 'blue', d)
            except:
                pass
            try:
                plotrect(ax[p], t[int(count2*nbins + e4[0])], t[int(count2*nbins + e4[2])],
                         flow[4, count], fhigh[4, count], 'red', d)
            except:
                pass
            count += 1
            count2 += 1

    fig.tight_layout()
    fig.show()
