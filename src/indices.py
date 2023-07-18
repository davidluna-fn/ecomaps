'''
Contiene los algoritmos que calculan cada uno de los posibles descriptores de paisaje acústico.

'''

import numpy as np
from scipy import signal
from maad import features
from scipy.stats import entropy


def spectral_centroid_spread(s, freq):
    m = np.repeat(freq.reshape(len(freq), 1), s.shape[1], 1)
    C = np.sum(s*m, axis=0) / np.sum(s, axis=0)
    k = m - np.repeat(C.reshape(1, len(C)), m.shape[0], 0)
    S = np.sum((k**2)*s, axis=0) / np.sum(s, axis=0)
    return C, S


def spectral_entropy(s):
    '''
        Calcula la entropia a lo largo del espectro
    '''
    s = np.divide(s, np.sum(s, axis=0))
    hf = entropy(s, axis=0)/np.log2(s.shape[0])
    return hf


def ACIft(s):
    '''

    Calcula el índice de complejidad acústica que se calcula primero sobre la frecuencia y luego sobre el tiempo (ACIft)
    [2]

    :param s: Espectrograma de la señal (numpy array)
    :return: el valor del ACIft (float)
    '''

    s = s / np.amax(s)
    ACI = np.sum(np.divide(np.absolute(
        np.diff(s, axis=0)), s[1:, :] + s[:-1, :]), axis=0)
    return ACI


def ADIm(s, Fs, wband=1000):
    '''
    Calcula el vector de ADI modificado propuesto en [4]

    :param s: Espectrograma de la señal (numpy array)
    :param Fs: Frecuencia de muestreo en Hz (int)
    :param wband: tamaño de cada banda de frecuencia en Hz, valor por defecto 1000 (int)
    :return: Un vector que contiene los valores del ADIm (numpy array)
    '''

    bn = background_noise_freq(s)

    # bn=-50
    # bn = 10**(bn/20)
    sclean = s - np.tile(bn, (s.shape[1], 1)).T
    sclean[sclean < 0] = 0
    sclean[sclean != 0] = 1
    Fmax = Fs/2
    nband = int(Fmax//wband)
    bin_step = int(s.shape[0]//nband)
    pbin = np.sum(sclean, axis=1)/s[:bin_step, :].size
    p = np.zeros(nband)

    for band in range(nband):
        p[band] = np.sum(pbin[band*bin_step:(band+1)*bin_step]) + 0.0000001

    ADIv = -np.multiply(p, np.log(p))
    return ADIv


def background_noise_freq(s):
    '''

    Calcula el valor del ruido de fondo para cada celda del espectrograma en el eje de las frecuencias [5]

    :param s: Espectrograma de la señal (numpy array)
    :return: Vector que contiene el valor del ruido de fondo para cada celda de frecuencia (numpy array)
    '''

    nfbins = s.shape[0]
    bn = np.zeros(nfbins)
    for i in range(nfbins):
        f = s[i, :]
        nbins = int(s.shape[1]/8)
        H, bin_edges = np.histogram(f, bins=nbins)
        fwin = 5
        nbinsn = H.size-fwin
        sH = np.zeros(nbinsn)

        for j in range(nbinsn):
            sH[j] = H[j:j+fwin].sum()/fwin

        modep = sH.argmax()
        mode = np.amin(f) + (np.amax(f)-np.amin(f))*(modep/nbins)

        acum = 0
        j = 0
        Hmax = np.amax(sH)
        while acum < 0.68*Hmax:
            acum += H[j]
            j += 1

        nsd = np.amin(f) + (np.amax(f)-np.amin(f))*(j/nbins)
        bn[i] = mode + 0.1*nsd
    return bn


def gini(x, corr=False):
    """
    Gini

    Compute the Gini value of x

    Parameters
    ----------
    x : ndarray of floats
        Vector or matrix containing the data

    corr : boolean, optional, default is False
        Correct the Gini value

    Returns
    -------  
    G: scalar
        Gini value

    References
    ----------
    Ported from ineq library in R
    """
    if sum(x) == 0:
        G = 0  # null gini
    else:
        n = len(x)
        x.sort()
        G = sum(x * np.arange(1, n+1, 1))
        G = 2 * G/sum(x) - (n + 1)
        if corr:
            G = G/(n - 1)
        else:
            G = G/n
    return G


def AEIm(s, Fs, wband=1000):
    '''
    Calcula el vector de ADI modificado propuesto en [4]

    :param s: Espectrograma de la señal (numpy array)
    :param Fs: Frecuencia de muestreo en Hz (int)
    :param wband: tamaño de cada banda de frecuencia en Hz, valor por defecto 1000 (int)
    :return: Un vector que contiene los valores del ADIm (numpy array)
    '''

    bn = background_noise_freq(s)
    # bn=-50
    # bn = 10**(bn/20)
    sclean = s - np.tile(bn, (s.shape[1], 1)).T
    sclean[sclean < 0] = 0
    sclean[sclean != 0] = 1
    Fmax = Fs/2
    nband = int(Fmax//wband)
    bin_step = int(s.shape[0]//nband)
    pbin = np.sum(sclean, axis=1)/s[:bin_step, :].size
    p = np.zeros(nband)

    for band in range(nband):
        p[band] = np.sum(pbin[band*bin_step:(band+1)*bin_step]) + 0.0000001

    print(p.shape)
    AEIv = gini(p)
    return AEIv


def beta(s, f, bio_band=(2000, 8000)):
    '''

    Calcula el índice bioacústico de la señal (β) [6]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band: tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :return: el valor de β (float)
    '''

    minf = bio_band[0]
    maxf = bio_band[1]
    s = s/np.amax(s)
    s = np.abs(10*np.log10(s**2))
    bioph = s[np.logical_and(f >= minf, f <= maxf), :]
    bioph_norm = bioph - np.amin(bioph, axis=0)
    B = 1 - np.sum(bioph_norm, axis=0)
    return B


def median_envelope(audio, Fs, depth=16):
    '''

    La mediana del envolvente de la amplitud (M)[9].

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :param depth: la profundidad de digitalización de la señal, valor por defecto 16 (int)
    :return: el valor de M (float)
    '''

    min_points = Fs*60
    npoints = len(audio)
    y = []
    VerParticion = npoints/min_points

    if (VerParticion >= 3):
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))
    else:
        if (VerParticion == 1):
            min_points = Fs*20
        else:
            min_points = Fs*30
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))

    y = np.concatenate([y])
    M = (2**(1-depth))*np.median(y)
    return M


def NDSI(s, f, bio_band=(2000, 8000), tech_band=(200, 1500)):
    '''

    Calcula el índice NDSI [12] que hace una relación entre el nivel de biofonía y tecnofonía. -1 indica biofonía pura y
    1 indica pura tecnofonía.

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band:  tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :param tech_band: tupla con la frecuencia mínima y máxima de la banda tecnofónica, valor por defecto: (200, 1500) (tuple)
    :return: el valor NDSI de la señal (float)
    '''
    # s = np.mean(s, axis=1)
    s = s ** 2

    bio = s[np.logical_and(f >= bio_band[0], f <= bio_band[1]), :]

    B = np.trapz(bio, f[np.logical_and(
        f >= bio_band[0], f <= bio_band[1])], axis=0)

    tech = s[np.logical_and(f >= tech_band[0], f <= tech_band[1]), :]
    A = np.trapz(tech, f[np.logical_and(
        f >= tech_band[0], f <= tech_band[1])], axis=0)
    ND = (A-B)/(B+A)
    return ND


def f1metric(s, f, bio_band=(2000, 8000), tech_band=(200, 1500)):
    '''

    Calcula el índice NDSI [12] que hace una relación entre el nivel de biofonía y tecnofonía. -1 indica biofonía pura y
    1 indica pura tecnofonía.

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param bio_band:  tupla con la frecuencia mínima y máxima de la banda biofónica, valor por defecto: (2000, 8000) (tuple)
    :param tech_band: tupla con la frecuencia mínima y máxima de la banda tecnofónica, valor por defecto: (200, 1500) (tuple)
    :return: el valor NDSI de la señal (float)
    '''
    # s = np.mean(s, axis=1)
    s = s ** 2

    bio = s[np.logical_and(f >= bio_band[0], f <= bio_band[1]), :]

    B = np.trapz(bio, f[np.logical_and(
        f >= bio_band[0], f <= bio_band[1])], axis=0)

    tech = s[np.logical_and(f >= tech_band[0], f <= tech_band[1]), :]
    A = np.trapz(tech, f[np.logical_and(
        f >= tech_band[0], f <= tech_band[1])], axis=0)
    ND = (B*A)/(B+A)
    return ND


def number_of_peaks(syy, f, nedges=10, slope=6, freq_dis=100):
    '''

    Cuenta el número de picos en el espectro medio de la señal [13].

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param nedges: número de partes en las que se divide la señal, por defecto 10 (int)
    :return: número de picos de la señal.
    '''

    # Filtro de media móvil
    def smooth(a, n=10):
        '''

        Esta función suaviza la señal con un filtro de media móvil.

        :param a: señal (numpy array)
        :param n: tamaño de la ventana del filtro de media móvil, por defecto, 10 (int)
        :return: señal suavizada
        '''

        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    NP_ = []
    for ind in range(syy.shape[1]):
        s = np.reshape(syy[:, ind], (syy[:, ind].shape[0], 1))
        NP = features.number_of_peaks(
            s, f, slopes=slope, min_freq_dist=freq_dis, display=False)

        NP_.append(NP)
    return np.array(NP_)


def spectral_maxima_entropy(s, f, fmin, fmax):
    '''

    Calcula la entropía de los máximos espectrales (Hm)[10]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param fmin: frecuencia inferior de la banda en la que se hará el análisis en Hz (int)
    :param fmax: frecuencia superior de la banda en la que se hará el análisis en Hz (int)
    :return: valor del Hm (float)
    '''

    s = s/np.amax(s)
    s_max = np.max(s, axis=1)
    s_band = s_max[np.logical_and(f >= fmin, f >= fmax)]
    s_norm = s_band/np.sum(s_band)
    N = len(s_norm)
    Hm = -np.sum(np.multiply(s_norm, np.log2(s_norm)))/np.log2(N)
    return Hm


def spectral_variance_entropy(s, f, fmin, fmax):
    '''

     Calcula la entropía de la varianza espectral (Hv)[10]

    :param s: Espectrograma de la señal (numpy array)
    :param f: vector de frecuencias correspondientes al espectrograma s (numpy array)
    :param fmin: frecuencia inferior de la banda en la que se hará el análisis en Hz (int)
    :param fmax: frecuencia superior de la banda en la que se hará el análisis en Hz (int)
    :return: valor del Hv (float)
    '''

    s = s/np.amax(s)
    s_std = np.std(s, axis=1)
    s_band = s_std[np.logical_and(f >= fmin, f >= fmax)]
    s_norm = s_band/np.sum(s_band)
    N = len(s_norm)
    Hv = -np.sum(np.multiply(s_norm, np.log2(s_norm)))/np.log2(N)
    return Hv


def temporal_entropy(audio, Fs):
    '''

    Calcula la entropía acústica temporal (Ht)[15]

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz (int)
    :return: el valor de Ht (float)
    '''

    min_points = Fs*60
    npoints = len(audio)
    y = []
    VerParticion = npoints/min_points

    if (VerParticion >= 3):
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))
    else:
        if (VerParticion == 1):
            min_points = Fs*20
        else:
            min_points = Fs*30
        for seg in range(min_points, npoints, min_points):
            y.append(np.abs(signal.hilbert(audio[seg - min_points:seg])))

    env = np.concatenate([y])
    env_norm = env/np.sum(env)

    N = len(env_norm)
    Ht = -np.sum(np.multiply(env_norm, np.log2(env_norm)))/np.log2(N)
    return Ht

    '''


    Referencias:

    [1] Pieretti, N., Farina, A., & Morri, D. (2011). A new methodology to infer the singing activity of an avian
        community: The Acoustic Complexity Index (ACI). Ecological Indicators, 11(3), 868–873.
        http://doi.org/10.1016/j.ecolind.2010.11.005

    [2] Farina, A., Pieretti, N., Salutari, P., Tognari, E., & Lombardi, A. (2016). The Application of the Acoustic
        Complexity Indices (ACI) to Ecoacoustic Event Detection and Identification (EEDI) Modeling. Biosemiotics, 9(2),
        227–246. http://doi.org/10.1007/s12304-016-9266-3

    [3] Pekin, B. K., Jung, J., Villanueva-Rivera, L. J., Pijanowski, B. C., & Ahumada, J. A. (2012). Modeling acoustic
        diversity using soundscape recordings and LIDAR-derived metrics of vertical forest structure in a neotropical
        rainforest. Landscape Ecology, 27(10), 1513–1522. http://doi.org/10.1007/s10980-012-9806-4

    [4] Duque-Montoya, D. C. (2018). Methodology for Ecosystem Change Assessing using Ecoacoustics Analysis.
        Universidad de Antioquia.

    [5] Towsey, M. (2013). Noise removal from waveforms and spectrograms derived from natural recordings of the
        environment. Retrieved from http://eprints.qut.edu.au/61399/

    [6] Boelman, N. T., Asner, G. P., Hart, P. J., & Martin, R. E. (2007). Multi-trophic invasion resistance in Hawaii:
        Bioacoustics, field surveys, and airborne remote sensing. Ecological Applications, 17(8), 2137–2144.
        http://doi.org/10.1890/07-0004.1

    [7] Torija, A. J., Ruiz, D. P., & Ramos-Ridao, a F. (2013). Application of a methodology for categorizing and
        differentiating urban soundscapes using acoustical descriptors and semantic-differential attributes.
        The Journal of the Acoustical Society of America, 134(1), 791–802. http://doi.org/10.1121/1.4807804

    [8] Tchernichovski, O., Nottebohm, F., Ho, C., Pesaran, B., & Mitra, P. (2000). A procedure for an automated
        measurement of song similarity. Animal Behaviour, 59(6), 1167–1176. http://doi.org/10.1006/anbe.1999.1416

    [9] Depraetere, M., Pavoine, S., Jiguet, F., Gasc, A., Duvail, S., & Sueur, J. (2012). Monitoring animal diversity
        using acoustic indices: Implementation in a temperate woodland. Ecological Indicators, 13(1), 46–54.
        http://doi.org/10.1016/j.ecolind.2011.05.006

    [10] Towsey, M., Wimmer, J., Williamson, I., & Roe, P. (2014). The use of acoustic indices to determine avian
        species richness in audio-recordings of the environment. Ecological Informatics, 21, 110–119.
        http://doi.org/10.1016/j.ecoinf.2013.11.007

    [11] De Coensel, B., Botteldooren, D., Debacq, K., Nilsson, M. E., & Berglund, B. (2007).
        Soundscape classifying ants. In Internoise. http://doi.org/10.1260/135101007781447993

    [12] Kasten, E. P., Gage, S. H., Fox, J., & Joo, W. (2012). The remote environmental assessment laboratory’s
        acoustic library: An archive for studying soundscape ecology. Ecological Informatics, 12, 50–67.
        http://doi.org/10.1016/j.ecoinf.2012.08.001

    [13] Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity Sampling Using a Global
        Acoustic Approach: Contrasting Sites with Microendemics in New Caledonia. PLoS ONE, 8(5), e65311.
        http://doi.org/10.1371/journal.pone.0065311

    [14] Qi, J., Gage, S. H., Joo, W., Napoletano, B., & Biswas, S. (2007). Soundscape characteristics of an
        environment: a new ecological indicator of ecosystem health. In Wetland and Water Resource Modeling and
        Assessment: A Watershed Perspective (Vol. 20071553, pp. 201–214). http://doi.org/10.1201/9781420064155

    [15] Sueur, J., Pavoine, S., Hamerlynck, O., & Duvail, S. (2008). Rapid Acoustic Survey for Biodiversity Appraisal.
        PLoS ONE, 3(12), e4065. http://doi.org/10.1371/journal.pone.0004065

    [16] Merchant, N. D., Fristrup, K. M., Johnson, M. P., Tyack, P. L., Witt, M. J., Blondel, P., & Parks, S. E.
        (2015). Measuring acoustic habitats. Methods in Ecology and Evolution, 6(3), 257–265.
        http://doi.org/10.1111/2041-210X.12330

    [17] Mitrović, D., Zeppelzauer, M., & Breiteneder, C. (2010). Features for Content-Based Audio Retrieval.
         Advances in Computers, 78(10), 71–150. http://doi.org/10.1016/S0065-2458(10)78003-7
    '''
