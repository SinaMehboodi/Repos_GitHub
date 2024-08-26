import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, to_numeric, read_csv
from scipy.fft import fft, fft2, fftfreq, ifft, ifft2
from matplotlib import ticker
from scipy.io import loadmat
from os import listdir
from os.path import exists
from lmfit import Model, fit_report
from time import time
from functools import wraps

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

def derivative_divide(

        X=None, Y=None, Z=None,
        axis:{
            "type": "int",
            "min":0, "max": 1,
            "hint": "Axis along which the difference quotient is calculated."
            }=0,
        modulation_amp:{
            "type": "int",
            "min": 0,
            "hint": "Perform difference quotient with this step width."
            }=1,
        average:{
            "type": "bool",
            "hint": "Average results of function with modulation_amp=range(0,modulation_amp+1).",
            }=True
        ):
        """
        Perform numerical derivative along the axis given by the axis argument and
        divide by the non-derivative value.

        As a background correction method this "dd" method helps in the case where
        the background enters as `V_o`:

        .. math:: S(f,H) = \\frac{V_o(f) + Z V_o(f) \chi(H)}{V_i} \\cdot \\exp(\\imath\,\\phi)

        as this results in

        .. math:: \mathrm{dd}(S(f,H)) = Z \\cdot \\frac{\\chi(H+\\Delta H)-\\chi(H-\\Delta H)}{\\Delta H} + O(Z^2)

        in the limit of small \Delta H which is equivalent to the partial
        derivative

        .. math:: \\frac{\mathrm{d}\\chi}{\mathrm{d}H} \cdot Z

        Most notably the phase φ which is, in spectroscopic measurements,
        given only by the electrical length and usually complicates data analysis,
        drops out and Z is a scalar real quantity.

        Furthermore a smoothing can be implemented by specifying modulation_amp

        Params:
        =======

        X : array_like (NxM)
            Input X-values (independent variable, M-axis)
        Y : array_like (NxM)
            Input Y-values (independent variable, N-axis)
        Z : array_like (NxM)
            Input Z-values (Signal, dependent variable)
        axis : {0,1}, optional (default: 0)
            The axis, along which the derivative is calculated
        modulation_amp : int, optional (default: 1)
            The number of steps over which the central difference is computed and
            averaged if average is True
        average : bool, optional (default: True)
            If set to True and modulation_amp > 1: Perform operation for 
            modulation_amp 0, 1, 2, 3,... and average the resulting values.
        References:
        ===========

        [1] Maier-Flaig et al. " Analysis of broadband ferromagnetic resonance in the frequency domain"
            arxiv preprint, arXiv:1705.05694 [https://arxiv.org/abs/1705.05694]

        """
        if axis == 0:
            delta = np.diff(X, axis=axis)
        elif axis == 1:
            Z = Z.T
            delta = np.diff(Y, axis=axis).T
        else:
            raise(ValueError("Only two dimensional datasets are supported"))

        G = np.zeros_like(Z)
        for row in np.arange(modulation_amp, np.shape(Z)[0]-modulation_amp):
            if average:
                zl = np.mean(Z[row-modulation_amp:row, :], axis=0)
                zh = np.mean(Z[row:row+modulation_amp+1, :], axis=0)
            else:
                zl = Z[row-modulation_amp, :]
                zh = Z[row+modulation_amp, :]
            zm = Z[row, :]
            d = np.mean(delta[row-modulation_amp:row+modulation_amp, :], axis=0)

            G[row, :] = (zh-zl)/zm/d

        if axis == 1:
            G = G.T

        return X, Y, G, d

def derivative(
        X=None, Y=None, Z=None,
        axis:{
            "type": "int",
            "min":0, "max": 1,
            "hint": "Axis along which the difference quotient is calculated."
            }=0,
        modulation_amp:{
            "type": "int",
            "min": 0,
            "hint": "Perform difference quotient with this step width."
            }=1
        ):
    if axis == 0:
        delta = np.diff(X, axis=axis)
    elif axis == 1:
        Z = Z.T
        delta = np.diff(Y, axis=axis).T
    else:
        raise(ValueError("Only two dimensional datasets are supported"))

    G = np.zeros_like(Z)
    for row in np.arange(modulation_amp, np.shape(Z)[0]-modulation_amp):
            zl = Z[row-modulation_amp, :]
            zh = Z[row, :]
            d = np.mean(delta[row-modulation_amp:row+modulation_amp, :], axis=0)
            G[row, :] = (zh-zl)/d

    if axis == 1:
        G = G.T

    return X, Y, G, d

def dS21(x, A, Psi, fres, Df, mod, Msat, H0): # no background needed?: slope_real=None,slope_imag=None, off_real=None, off_imag=None,
        """ x corresponds to frequency f in GHz
            mod in units of Tesla
        See Eq. (5) from https://aip.scitation.org/doi/pdf/10.1063/1.5045135?class=pdf"""

        mu_0 = 4.0*np.pi*1e-7
        gamma = 1.76085963023e11 # gyromagnetic ratio [rad*1/(s*T)]

        omega = 2*np.pi*x
        modfield = mod
        omegaRes= 2*np.pi*fres
        Domega= 2*np.pi*Df

        modOmega = modfield * gamma * mu_0 # modOmega = modH * d omega/ d H0 -> assume linear "dispersion" -> derivative becomes "gamma * mu0"
        Chiplus = gamma*mu_0*Msat * (gamma*mu_0*H0 - 1j * Domega)/(omegaRes**2 - (omega+modOmega)**2 - 1j * (omega+modOmega) *Domega)
        Chiminus = gamma*mu_0*Msat * (gamma*mu_0*H0 - 1j * Domega)/(omegaRes**2 - (omega-modOmega)**2 - 1j * (omega-modOmega) *Domega)

        #background = off_real + slope_real*omega + 1j*(off_imag + slope_imag*omega)

        #fitting model as in eq. (5)`
        model = np.conjugate(-1j* omega * A * np.exp(1j * Psi) * (Chiplus-Chiminus)/(2* modOmega)) #+ background

        return np.real(model)

def Lorentzian(x, A, Psi, fres, Df, mod, Msat, Ho):

    ####################################################################################
    ####################################################################################
    #                                                                                  #
    #   Implement me                                                                   #
    #                                                                                  #
    ####################################################################################
    ####################################################################################

    print("Implement me!")

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

class Spectrum:
    
    def __init__(self, path, saturation=None, skip_field=1, skip_freq=1, derivative_divide=True , warning=True,modulation_amp=1):
        """
        :param path: the filepath of the measurement data as generated by LabView
        :param saturation: the no. of san where the Kittel mode saturates
        :param skip_field: takes into account only every -skip- element of the field range. Useful when dealing with large datasets.
        :param skip_freq: takes into account only every -skip- element of the frequency range. Useful when dealing with large datasets.
        :derivative_divide: if False the real spectrum will be shown. If using the builtin plot function will probably need to readjust the vmin and vmax values to get a better plot.
        :param warning: if set to true will warn the user about not setting the no. of scan where the Kittel mode saturates
        """
        self.modulation_amp=modulation_amp
        self.path = path
        self.skip_field = skip_field
        self.skip_freq = skip_freq
        self.derivative_divide = derivative_divide
        self.B, self.Field, self.Freq, self.S21dd = self.get()
        self.saturation = saturation

        if self.saturation == None:
            
            if warning:

                print("The scan no. where the kittel mode ends was not chosen. All fits called from this object will treat Msat as a free parameter. This might decrease fitting accuracy.\
                It is strongly suggested to call the plot method of this object, use it to establish the saturtion no. of scan, and then reinitialize the object with the corresponding saturation value.")
 
            self.Msat = None

        else:

            self.Msat = self.B[np.where(self.B==self.B[saturation])[0][0]] 

    def get(self, skip_field=1, skip_freq=1):
        """
        Retrieves all the relevant data about the spectrum.

        :param skip_field: takes into account only every -skip- element of the field range. Useful when dealing with large datasets. If a spectrum object has been initialized with a skip_field value this will skip the
        datapoints of the already reduced dataset.
        :param skip_freq: takes into account only every -skip- element of the frequency range. Useful when dealing with large datasets. If a spectrum object has been initialized with a skip_field value this will skip the
        datapoints of the already reduced dataset.

        :return: B - an array of fields (1D),
        Field - an array of column numbers (2D),
        Freq - an array of frequencies (2D),
        S21 - an array of spectrum intensities after derivative divide has been applied (2D) if derivative_divide was left at the default of True while initializing the Spectrum object, or an array of spectrum intensities
        after background has been substracted if derivatie_divide is set to False.
        """ 
        
        f1, Freq, S21Refmag, S21Mag = self.data_loader()
        
        S21Mag=S21Mag[::self.skip_freq*skip_freq, ::self.skip_field*skip_field]
        Freq=Freq[::self.skip_freq*skip_freq, ::self.skip_field*skip_field]
        
        ####################################
        # Other possible dataloader outputs#     
        ####################################                              
        #S21Imag=S21Imag1["Data"][:,:]     #
        #Mlog=Mlog1["Data"][:,:]           #
        #Phase=Phase1["Data"][:,:]         #
        #S21Real=S21Real1["Data"][:,:]     #
        ####################################

        Field = np.arange(0, len(Freq[0,:]), 1)
        repetitions = len(Freq)
        Field = np.transpose([Field] * repetitions)

        field0 = f1["field"][1::1]
        field1 = to_numeric(field0, errors="coerce")  
        field1=field1.to_numpy()

        B = field1[::self.skip_field*skip_field]

        if self.derivative_divide:

            Field, Freq, S21dd, __ = derivative_divide(X=Field,Y=Freq,Z=S21Mag.T, modulation_amp=self.modulation_amp, axis=0)

            return B, Field, Freq, S21dd

        else:
            
            Field, Freq, S21d , __ = derivative(X=Field,Y=Freq,Z=S21Mag.T, modulation_amp=self.modulation_amp, axis=0)
            #For ref diff 
            # S21Refmag=S21Refmag[::self.skip_freq*skip_freq, ::self.skip_field*skip_field]
            # S21_Dif =S21Mag-S21Refmag
            # S21d=S21_Dif.T
            
            return B, Field, Freq, S21d

    @timeit
    def scatter_plot(self, field_cuts_list, frequencies_list, frequencies_end_list = None, colors = None, denoise = False, save_name = None, v_min=-0.005, v_max=0.005,c_map="PuOr",style="Beam",pic_size=240,f_lim=None,s_lim=None,nom_locator=50):
        """
        Creates a figure of the FMR spectrum with points correponding to the resonance frequencies of the fitted anti-lorentzian peaks.
        
        :param field_cuts: a list/tuple of rownumbers (i.e. [arange(33, 36, 1), arange(40, 45), ...]). These elements five the row number range in which the given instance of fitting
        will occur
        :param frequencies_list: a list/tuple of two element lists/tuples containing the minimum and maximum frequencies freq_min and freq_max (i.e. [[fmin1, fmax1], [fmin2, fmax2], ...]).
        These frequencies give the frequency range in which the given instance of fitting will occur.
        :param frequencies_end_list: a list/tuple of two element lists/tuples containing the minimum and maximum frequencies freq_min and freq_max(i.e. [[fmin1, fmax1], [fmin2, fmax2], ...]).
        These frequencies give the frequency range in which the given instance of fitting will end. If these values are entered the linecuts will be taken at a linear segment between frequencies_list elements and frequencies_end list elements.
        :param colors: a list of strings signifying the shape and color of the plotted scatter points of the fit (i.e., ['ro', 'gs'] for a two separate fits having the first fit be red circles and the second green squares).
        If left as None the default is 'ro' (red circles) for all fits.
        :param denoise: if True denoises the spectrum with the help of a Fourier transform and a low pass filter.
        :param save_name: names of the file the figure will be saved as. NEEDS TO END IN FILE EXTENSION (i.e. ___.jpg, ___.png, ...)!
        :param v_min: defines the vmin value in the pcolor plot. Change for possibly better contrast.
        :param v_max: defines the vmax value in the pcolor plot. Change for possibly better contrast. 
        :pic_size: if set (a,b) the figsize will be in inches as normal plt figure, and if set a single digit "A" (linewidth) the length of figure will be the A (pt) in latex format and the pic_size will be calculated by set_size finction 
        :return: a plot and a savefile
        """ 

        assert len(frequencies_list) == len(field_cuts_list), "The frequencies and field_cuts_list need to be of the same length!"

        for i in range(len(frequencies_list)):
            
            assert len(frequencies_list[i]) == 2, "Each element of the frequencies list should contain a sublist of two values, one for the minimum frequency (freq_min), and the other for maximum frequency (freq_max)!"

        if colors == None:
            
            colors = ['ro']*len(field_cuts_list)

        else:

            assert len(colors) == len(frequencies_list), "The colors list needs to be of the same length as the frequencies and field_cuts_list lists!"
        if isinstance(pic_size,tuple):
            pic_s=pic_size
        else:
            pic_s=set_size(pic_size)        
        fig, (ax1, ax) = plt.subplots(nrows=2,sharex=True, figsize=pic_s ,gridspec_kw={'height_ratios': [2, 6]})
        # Using seaborn's style
        plt.style.use(style)

        for i in range(len(field_cuts_list)):
        
            frequencies = (frequencies_list[i][0], frequencies_list[i][1])
            
            if frequencies_end_list == None:
            
                frequencies_end = None

            else:
                
                if frequencies_end_list[i] == None:

                    frequencies_end = None
                else:

                    frequencies_end = (frequencies_end_list[i][0], frequencies_end_list[i][1])
            
            field_cuts = field_cuts_list[i]

            fd_reso, fr_reso,__,__= self.scatter_fit(field_cuts, frequencies, frequencies_end)
            ax.plot(fd_reso, fr_reso, colors[i])
        
        ax1.plot(self.Field[:,0], self.B, c="blue")

        if denoise:

            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            
            im = ax.pcolor(Field, Freq.T, S21dd[:,:],
                       vmin=v_min, vmax=v_max, cmap=c_map,shading='auto')

        else:
        
            im = ax.pcolor(self.Field, self.Freq.T, self.S21dd[:,:],
                       vmin=v_min, vmax=v_max, cmap=c_map,shading='auto')

        cb_ax = fig.add_axes([1.01, 0.135, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cb_ax,ticks=[v_min,0,v_max])
        cbar.set_label("$Re(\partial_DS_{21}/\partial H)$")
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if self.derivative_divide==False:
            cbar.set_label("$\Delta S_{21}$")
        second = ax1.secondary_xaxis("top")
        second.set_xticks(self.Field[:,0], self.B, c="blue")#(self.B).astype(int)
        second.xaxis.set_major_locator(ticker.MultipleLocator(nom_locator))
        ax1.yaxis.set_major_locator(ticker.FixedLocator([min(self.B),
                                                         max(self.B)]))
        ax1.set_ylabel("$B$ (mT)")
        # ax1.grid()
        ax.set_xlabel("No. of Scan")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylabel("$f$ (GHz)")
        ax.set_ylim(min(self.Freq[:,0]),max(self.Freq[:,0]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        if s_lim!= None:
            ax.set_xlim(s_lim[0],s_lim[1])
        if f_lim!=None:
            ax.set_ylim(f_lim[0],f_lim[1])
        # fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        
        if save_name != None:

            plt.savefig(save_name, bbox_inches='tight')

    @timeit
    def plot(self, save_name=None, denoise = False, v_min=-0.001, v_max=0.001,c_map="bone",pic_size=480,style="Beam",nom_locator=20):
        """
        Creates a figure of the FMR spectrum.

        :param save_name: names of the file the figure will be saved as. NEEDS TO END IN FILE EXTENSION (i.e. ___.jpg, ___.png, ...)!
        :param denoise: if True denoises the spectrum with the help of a Fourier transform and a low pass filter.
        :param v_min: defines the vmin value in the pcolor plot. Change for possibly better contrast.
        :param v_max: defines the vmax value in the pcolor plot. Change for possibly better contrast.

        :return: a plot and a savefile
        """ 
        # Using seaborn's style
        plt.style.use(style)

        if isinstance(pic_size,tuple):
            pic_s=pic_size
        else:
            pic_s=set_size(pic_size)

        fig, (ax1, ax) = plt.subplots(nrows=2, sharex=True, figsize=pic_s ,gridspec_kw={'height_ratios': [2, 6]})
        
        ax1.plot(self.Field[:,0], self.B, c="blue")
        
        if denoise:
            
            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            
            im = ax.pcolor(Field, Freq.T, S21dd[:,:],
                       vmin=-0.001, vmax=0.001, cmap=c_map,shading='auto')

        else:
        
            im = ax.pcolor(self.Field, self.Freq.T, self.S21dd[:,:],
                       vmin=v_min, vmax=v_max, cmap=c_map,shading='auto') #vmin=-0.05, vmax=0.05
        
        cb_ax = fig.add_axes([1.01, 0.135, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cb_ax,ticks=[v_min,0,v_max])
        cbar.set_label("$Re(\partial_DS_{21}/\partial H)$")
        cbar.formatter.set_powerlimits((0, 0))
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if self.derivative_divide==False:
            cbar.set_label("$\Delta S_{21}$")
        cbar.ax.tick_params()
        second = ax1.secondary_xaxis("top")
        second.set_xticks(self.Field[:,0],self.B, c="blue")# (self.B).astype(int), c="blue", size=20)
        second.xaxis.set_major_locator(ticker.MultipleLocator(nom_locator))
        ax1.yaxis.set_major_locator(ticker.FixedLocator([min(self.B),
                                                         max(self.B)]))
        ax1.set_ylabel("$B$ (mT)")
        # ax1.grid()
        ax.set_xlabel("No. of Scan")
        ax.set_ylabel("$f$ (GHz)")
        ax.set_xlim(0,len(self.Field[:,0]))
        ax.set_ylim(min(self.Freq[:,0]),max(self.Freq[:,0]))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        # fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_name != None:

            plt.savefig(save_name, bbox_inches='tight',dpi=900)

    @timeit
    def zoom_plot(self, freq_range, scan_range, save_name=None, denoise = False, v_min=-0.001, v_max=0.001,c_map="PuOr",pic_size= 480,style="Beam",unit=100,nom_locator=5):
        """
        Plots a square cut of the spectrum as defined by freq_range and scan_range.

        :param freq_range: defines the frequency range in which the square cut of the spectrum is to be taken.
        :param scan_range: defines the scan range (column number) in which the square cut of the spectrum is to be taken.
        :param v_min: defines the vmin value in the pcolor plot. Change for possibly better contrast.
        :param v_max: defines the vmax value in the pcolor plot. Change for possibly better contrast.

        :return: a plot of the zoomed in segment of the spectrum
        """
        
        freq_min = freq_range[0]
        freq_max = freq_range[1]
        scan_min = scan_range[0]
        scan_max = scan_range[1]

        f_min = np.where(self.Freq>freq_min)[0][0]
        f_max = np.where(self.Freq<freq_max)[0][-1]

        s_min = np.where(self.Field>scan_min)[0][0]
        s_max = np.where(self.Field<scan_max)[0][-1]

        # Using  style
        plt.style.use(style)
        if isinstance(pic_size,tuple):
            pic_s=pic_size
        else:
            pic_s=set_size(pic_size)
        fig, (ax1, ax) = plt.subplots(nrows=2, sharex=True, figsize=pic_s ,gridspec_kw={'height_ratios': [2, 6]})
        
        ax1.plot(self.Field[scan_min:scan_max,0], self.B[scan_min:scan_max], c="blue")
        ######
        if denoise:
            
            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            
            im = ax.pcolor(Field, Freq.T, S21dd[:,:],
                       vmin=-0.001, vmax=0.001, cmap=c_map,shading='auto')

        else:
        
            im = ax.pcolor(self.Field[scan_min:scan_max, f_min:f_max], self.Freq[f_min:f_max, scan_min:scan_max].T, self.S21dd[scan_min:scan_max, f_min:f_max],
                vmin=v_min, vmax=v_max, cmap=c_map,shading='auto')
        ######
        cb_ax = fig.add_axes([1.01, 0.135, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cb_ax,ticks=[v_min,0,v_max])
        cbar.set_label("$Re(\partial_DS_{21}/\partial H)$")
        cbar.formatter.set_powerlimits((0, 0))
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if self.derivative_divide==False:
            cbar.set_label("$\Delta S_{21}$")
        cbar.ax.tick_params()
        second = ax1.secondary_xaxis("top")
        ax1.set_ylabel("$B$ (mT)")
        ax.set_xlabel("No. of Scan")
        ax.set_ylabel("$f$ (GHz)")
        
        # second.set_xticks(self.Field[:,0],self.B, c="blue")# (self.B).astype(int), c="blue", size=20)
        second.xaxis.set_major_locator(ticker.MaxNLocator((scan_max-scan_min)/nom_locator))
        second.set_xticks([])# (self.B).astype(int), c="blue", size=20)
        ax1.yaxis.set_major_locator(ticker.FixedLocator([min(self.B[scan_min:scan_max]),
                                                         (min(self.B[scan_min:scan_max])+max(self.B[scan_min:scan_max]))/2,
                                                         max(self.B[scan_min:scan_max])]))
       
        ax1.set_yticks([min(self.B[scan_min:scan_max]),(min(self.B[scan_min:scan_max])+max(self.B[scan_min:scan_max]))/2, max(self.B[scan_min:scan_max])],[min(self.B[scan_min:scan_max])*unit,(min(self.B[scan_min:scan_max])+max(self.B[scan_min:scan_max]))/2*unit, max(self.B[scan_min:scan_max])*unit])
        ax1.grid()
        ax.xaxis.set_major_locator(ticker.MaxNLocator((scan_max-scan_min)/nom_locator))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylim(freq_min,freq_max)
        ax.set_xlim(scan_min,scan_max)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_name != None:

            plt.savefig(save_name, bbox_inches='tight',dpi=900)

    def scatter_fit(self, field_cuts, frequencies,  frequencies_end = None):
        """
        Carries out a fit for a given frequency range (freq_min to freq_max) at given field cuts (field_cuts).
    
        :param field_cuts: a list/tuple of column number cuts at which the fitting will occur for a given frequency range (i.e. arange(33, 36, 1) to fit the curves at column numbers from 33 to 35).
        :param frequencies: a tuple containing the lower and upper bounds of the frequency range where the scatter fitting will start. If frequencies_end is None the endpoints of the fit will also be taken to be in the same range.
        :param frequencies_end: a tuple containing the lower and upper bounds of the frequency range where the scatter fitting will end. Linecuts will be considered in a frequency range whose upper and lower bound are taken to be on the
        linear interval between frequencies and frequencies_end values 

        :return: fd_reso - a list containing all the fieldcut column numbers,
        fr_reso - a list containing resonance frequencies for given fieldcuts
        """ 

        fr_reso = []
        fd_reso = []
        ###
        Df_reso = []
        fd_stdr = []
        Df_stdr = []
        ###
        freq_min = frequencies[0]
        freq_max = frequencies[1]

        resonance = (freq_min + freq_max)/2

        if frequencies_end == None:
        
            ftop = np.linspace(frequencies[1], frequencies[1], len(field_cuts))
            fbot = np.linspace(frequencies[0], frequencies[0], len(field_cuts))

        else:

            ftop = np.linspace(frequencies[1], frequencies_end[1], len(field_cuts))
            fbot = np.linspace(frequencies[0], frequencies_end[0], len(field_cuts))

        for fit_num, cut in enumerate(field_cuts):
            
            fd_reso.append(cut)

            if field_cuts[0] < field_cuts[1]:

                linecut = Linecut(cut, self, (fbot[cut-np.min(field_cuts)], ftop[cut-np.min(field_cuts)]))
            
            else:

                linecut = Linecut(cut, self, (fbot[cut-np.min(field_cuts)-1], ftop[cut-np.min(field_cuts)-1]))

            if fit_num == 0:

                __ , __, fit_params,report = linecut.one_peak([resonance])
            
            else:

                __, __, fit_params,report = linecut.one_peak([resonance], fit_params)

            fr_reso.append(fit_params[2])
            Df_reso.append(fit_params[3])
            Df_stdr.append(report.params["m1_Df"].stderr)
            

        return fd_reso, fr_reso , Df_reso, Df_stdr
    
    def data_loader(self):
        """
        Helper function that loads the data from the files saved by the LabView VI. In the case that the data is stored in a different way only the dataloader needs to be changed to accomodate this.
        At the moment the data is saved according to Aisha's VI and needs to be combined using combine++ in MCQST/Equipment in order to work with this code.
        """
        MeasPath=self.path

        if not exists(MeasPath+ '\\' + 'MagneticField.csv'):
            
            listSubDir = listdir(MeasPath + r"\measurement\MEAS")
            main_Directory = []

            for i in np.arange(0,len(listSubDir)):

                main_Directory.append(MeasPath + r"\measurement\MEAS" + '\\' + listSubDir[i])
            
            ListFiles = listdir(main_Directory[0])
            ListFilesfreq = list(filter(lambda x: '__freq.txt' in x, ListFiles))

            MagneticField = []
            FieldName = []

            for i in np.arange(0,len(ListFilesfreq)):

                FileName = ListFilesfreq[i].split("_")
                FieldName.append(FileName[2])
                FieldN = float(FileName[2].replace("mT",""))
                MagneticField.append(FieldN)
            
            MagneticField = DataFrame(MagneticField, columns=['Magnetic Field Measure'])
            print('Magnetic field list is saved in This Address:\n',MeasPath)
            MagneticField.to_csv(MeasPath+ '\\' + 'MagneticField.csv')
        
        ###################################################################################
        # Other useful data that the dataloader could pull.                               #
        ###################################################################################
        #Mlog1 =scipy.io.loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\MLOG.mat")    #
        #Phase1 =scipy.io.loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\PHAS.mat")   #
        #S21Real1 = scipy.io.loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\REAL.mat")#
        #S21Imag1 = scipy.io.loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\IMAG.mat")#
        ###################################################################################

        Freq1 = loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\freq.mat")  
        S21Mag1 = loadmat(MeasPath+"\\"+r"measurement\store\MEAS\S21\MLIN.mat") 
        Refmag = loadmat(MeasPath+"\\"+r"measurement\store\REF\S21\MLIN.mat")
        
        f1 = read_csv(self.path+"\\MagneticField.csv",names=["num","field"])

        S21Mag=S21Mag1["Data"]
        Freq=Freq1["Data"]
        S21Refmag=Refmag["Data"]

        return f1, Freq, S21Refmag, S21Mag      
    
    @timeit
    def denoise_spectrum(self, fnoise=10):
        """
        Denoises the spectrum with the help of a Fourier transform and a low pass filter.

        :return: S21dd_denoised - an array S21dd_denoised (2D).
        """

        S21dd_denoised = np.zeros(np.shape(self.S21dd))

        Zfft = fft2(self.S21dd)
        high_freq_fft = Zfft.copy()

        for i in range(len(S21dd_denoised[:,0])):
            
            sample_freq = fftfreq(self.S21dd[:,i].size, d=0.001)
            high_freq_fft[np.where(fnoise<np.abs(sample_freq)), i]=0
        
        S21dd_denoised = ifft2(high_freq_fft)

        return S21dd_denoised[:,:-1]

    def __add__(self, other_spectrum):
        """
        Adds the 2D intensities arrays S21dd for both spectra.

        :param other_spectrum: the other spectrum whose S21dd values will be added to those of this instance of the class.

        :return: spectrum_result - the resulting spectrum, whose S21dd is the sum of self and other_spectrum.
        """
        
        assert np.shape(self.S21dd) == np.shape(other_spectrum.S21dd), "The two spectra need to be of the same shape in order to do operations with them!!"

        spectrum_result = Spectrum(path=self.path, saturation=self.saturation, skip_field=self.skip_field, skip_freq=self.skip_freq, derivative_divide=self.derivative_divide , warning=False)
        spectrum_result.S21dd = self.S21dd + other_spectrum.S21dd

        return spectrum_result
        
    def __sub__(self, other_spectrum):
        """
        Subtracts the 2D intensities arrays S21dd for both spectra.

        :param other_spectrum: the other spectrum whose S21dd values will be substracted from those of this instance of the class.

        :return: spectrum_result - the resulting spectrum, whose S21dd is the difference of self and other_spectrum.
        """

        assert np.shape(self.S21dd) == np.shape(other_spectrum.S21dd), "The two spectra need to be of the same shape in order to do operations with them!!"

        spectrum_result = Spectrum(path=self.path, saturation=self.saturation, skip_field=self.skip_field, skip_freq=self.skip_freq, derivative_divide=self.derivative_divide , warning=False)
        spectrum_result.S21dd = self.S21dd - other_spectrum.S21dd

        return spectrum_result

    def __mul__(self, num):
        """
        Multiplies the S21dd of this instance of the Spectrum by a number num.

        :param num: the number that multiplies this instance of the Spectrum class.

        :return: spectrum_result - the resulting spectrum, whose S21dd is multiplied by num.
        """

        spectrum_result = Spectrum(path=self.path, saturation=self.saturation, skip_field=self.skip_field, skip_freq=self.skip_freq, derivative_divide=self.derivative_divide , warning=False)
        spectrum_result.S21dd = spectrum_result.S21dd * num

        return spectrum_result

    def __truediv__(self, num):
        """
        Divides the S21dd of this instance of the Spectrum by a number num.

        :param num: the number that divides this instance of the Spectrum class.

        :return: spectrum_result - the resulting spectrum, whose S21dd is divided by num.
        """

        spectrum_result = Spectrum(path=self.path, saturation=self.saturation, skip_field=self.skip_field, skip_freq=self.skip_freq, derivative_divide=self.derivative_divide , warning=False)
        spectrum_result.S21dd = self.S21dd / num

        return spectrum_result
    
    
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    
    def zoom_cycle_plot(self,cycle_range,freq_range, scan_range, save_name=None,cycling=False ,denoise = False, v_min=-0.001, v_max=0.001,c_map="PuOr",pic_size= 480,style="Beam",unit=100,nom_locator=5,colorbar_cord=[1.01, 0.135, 0.02, 0.6]):
        """
        Plots a square cut of the spectrum as defined by freq_range and scan_range.

        :param freq_range: defines the frequency range in which the square cut of the spectrum is to be taken.
        :param scan_range: defines the scan range (column number) in which the square cut of the spectrum is to be taken.
        :param v_min: defines the vmin value in the pcolor plot. Change for possibly better contrast.
        :param v_max: defines the vmax value in the pcolor plot. Change for possibly better contrast.

        :return: a plot of the zoomed in segment of the spectrum
        """
        cycle_range_min=cycle_range[0]
        cycle_range_max=cycle_range[1]
        index_off=cycle_range_max-cycle_range_min
        freq_min = freq_range[0]
        freq_max = freq_range[1]
        scan_min = scan_range[0]
        scan_max = scan_range[1]

        f_min = np.where(self.Freq>freq_min)[0][0]
        f_max = np.where(self.Freq<freq_max)[0][-1]

        s_min = np.where(self.Field>scan_min)[0][0]
        s_max = np.where(self.Field<scan_max)[0][-1]

        # Using  style
        plt.style.use(style)
        if isinstance(pic_size,tuple):
            pic_s=pic_size
        else:
            pic_s=set_size(pic_size)
        fig, (ax1, ax) = plt.subplots(nrows=2, sharex=True, figsize=pic_s,gridspec_kw={'height_ratios': [2, 6]})
        if cycling:
            new_s21dd = np.concatenate((self.S21dd[scan_min:cycle_range_min,f_min:f_max],self.S21dd[cycle_range_max:scan_max,f_min:f_max]))
            new_array = np.concatenate((self.Field[scan_min:cycle_range_min,f_min:f_max],self.Field[cycle_range_max:scan_max,f_min:f_max]))
            new_Freq = np.concatenate((self.Freq[f_min:f_max,scan_min:cycle_range_min],self.Freq[f_min:f_max,cycle_range_max:scan_max]),axis=1)
            n_rows=new_array.shape[0]
            n_col=new_array.shape[1]
            new_Field = np.tile(np.arange(n_rows).reshape(n_rows, 1), (1, n_col))
            new_B_1D=np.concatenate((self.B[scan_min:cycle_range_min],self.B[cycle_range_max:scan_max]))
            
            ax1.plot(new_Field[:,0], new_B_1D, c="blue")
        else:
            ax1.plot(self.Field[scan_min:scan_max,0], self.B[scan_min:scan_max], c="blue")
        ######
        if denoise:
            
            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            
            im = ax.pcolor(Field, Freq.T, S21dd[:,:],
                       vmin=-0.001, vmax=0.001, cmap=c_map,shading='auto')

        else:
            if cycling:    
                im = ax.pcolor(new_Field, new_Freq.T, new_s21dd, vmin=v_min, vmax=v_max, cmap=c_map,shading='auto')
                
            else:
                im = ax.pcolor(self.Field[scan_min:scan_max, f_min:f_max], self.Freq[f_min:f_max, scan_min:scan_max].T, self.S21dd[scan_min:scan_max, f_min:f_max],
                    vmin=v_min, vmax=v_max, cmap=c_map,shading='auto')
        ######
        [1.01, 0.135, 0.02, 0.6]
        colorbar_a =colorbar_cord[0]
        colorbar_b =colorbar_cord[1]
        colorbar_c =colorbar_cord[2]
        colorbar_d =colorbar_cord[3]
        cb_ax = fig.add_axes([colorbar_a, colorbar_b, colorbar_c, colorbar_d])
        cbar = fig.colorbar(im, cax=cb_ax,ticks=[v_min,0,v_max])
        cbar.set_label("$Re(\partial_DS_{21}/\partial H)$")
        cbar.formatter.set_powerlimits((0, 0))
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if self.derivative_divide==False:
            cbar.set_label("$\Delta S_{21}$")
        cbar.ax.tick_params()
        second = ax1.secondary_xaxis("top")
        ax1.set_ylabel("$B$ (mT)")
        ax.set_xlabel("No. of Scan")
        ax.set_ylabel("$f$ (GHz)")
        
        # second.set_xticks(self.Field[:,0],self.B, c="blue")# (self.B).astype(int), c="blue", size=20)
        second.xaxis.set_major_locator(ticker.MaxNLocator((scan_max-index_off-scan_min)/nom_locator))
        second.set_xticks([])# (self.B).astype(int), c="blue", size=20)
        ax1.yaxis.set_major_locator(ticker.FixedLocator([min(self.B[scan_min:scan_max-index_off]),
                                                         (min(self.B[scan_min:scan_max-index_off])+max(self.B[scan_min:scan_max-index_off]))/2,
                                                         max(self.B[scan_min:scan_max-index_off])]))
       
        ax1.set_yticks([min(self.B[scan_min:scan_max]),(min(self.B[scan_min:scan_max])+max(self.B[scan_min:scan_max]))/2, max(self.B[scan_min:scan_max])],[min(self.B[scan_min:scan_max])*unit,(min(self.B[scan_min:scan_max])+max(self.B[scan_min:scan_max]))/2*unit, max(self.B[scan_min:scan_max])*unit])
        ax1.grid()
        ax.xaxis.set_major_locator(ticker.MaxNLocator((scan_max-index_off-scan_min)/nom_locator))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylim(freq_min,freq_max)
        ax.set_xlim(scan_min,scan_max-index_off)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        fig.tight_layout()
        
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_name != None:

            plt.savefig(save_name, bbox_inches='tight',dpi=900)
    
    
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    
  

class Linecut:
    
    def __init__(self, cut, Spectrum, frequencies = (0, 20e9)):
        """
        :param cut: the column number where the constant field cut is taken
        :param Spectrum: an instance of the Spectrum class. The linecuts will be taken from this spectrum.
        :param frequencies: a tuple containing the lower and upper bounds of the frequency range considerd in the instance of a linecut (leave default values for the whole linecut)
        """    

        self.cut = cut
        self.frequencies = frequencies
        self.B = Spectrum.B
        self.Field = Spectrum.Field
        self.Freq = Spectrum.Freq
        self.S21dd = Spectrum.S21dd
        self.derivative_divide = Spectrum.derivative_divide
        self.Freq_cut, self.S21dd_cut = self.get_cut()
        self.H0 = self.B[cut]
        self.Msat = Spectrum.Msat

    def get_cut(self,denoise=False):
        """
        Obtains the fieldcut segment data.

        :return: Freq_cut - an array (1D) containing all the frequencies for a given fieldcut
        S21dd_cut - an array (1D) containing all the derivative divide intensities of a given fieldcut,
        """    

        freq_min = self.frequencies[0]
        freq_max = self.frequencies[1]

        f_min = np.where(self.Freq>freq_min)[0][0]
        f_max = np.where(self.Freq<freq_max)[0][-1]

        Freq_cut = self.Freq[f_min:f_max, self.cut]
        S21dd_cut = self.S21dd[self.cut, f_min:f_max]
        if denoise:  
            __, S21dd_cut = self.denoising()
            
        return Freq_cut, S21dd_cut

    @timeit
    def plot(self, save_name=None, resonances=None, denoise=False,style="Beam",pic_size=240):
        """
        Plots the linecut segment of the spectrum.

        :param save_name: names of the file the figure will be saved as. NEEDS TO END IN FILE EXTENSION (i.e. ___.jpg, ___.png, ...)!
        :param resonances: if None the field cut segment will be plotted, but not fitted. If it is a list/tuple of up to three elements
         containing the information about where the resonance frequencies are expected to be, the fieldcut segment will be fitted for the
         given peaks (i.e. list containing one resonance -> one peak fit, list containing two resonances -> two peak fit, ...)
        :param denoise: boolean if true another figure of the denoised data will be plotted and saved under the name "denoised_" + save_name

        :return: a plot and a savefile
        """     
        plt.style.use(style)
        fig = plt.figure(figsize=set_size(pic_size))
            
        plt.plot(self.Freq_cut*1e-9, self.S21dd_cut, label = "real data")

        if resonances != None:

            num_reso = len(resonances)

            if num_reso == 1:
                
                __ , our_fit, __ ,details= self.one_peak(resonance = resonances, details = True)
                plt.plot(self.Freq_cut*1e-9, our_fit, label = "out best fit of data")

            
            elif num_reso == 2:
                
                __, our_fit, __ = self.two_peak(resonances, details=True)
                plt.plot(self.Freq_cut*1e-9, our_fit, label = "out best fit of data")

            elif num_reso == 3:
                
                __, our_fit, __ = self.three_peak(resonances, details=True)
                plt.plot(self.Freq_cut*1e-9, our_fit, label = "out best fit of data")
                
            elif num_reso > 3:
                
                print("At most three peaks can be selected for fitting!")
            
            else:

                print("The resonances variable needs to be an iterable object that returns number values (i.e. a tuple, list, array...)!")

        plt.xlabel("$f$ (GHz)")
        plt.ylabel("Intensity")
        plt.legend()
        # fig.show()

        if save_name != None:

            plt.savefig(save_name, bbox_inches='tight')

        if denoise:  
            
            __, S21dd_denoised_cut = self.denoising()

            fig = plt.figure(figsize=set_size(pic_size))
            plt.plot(self.Freq_cut*1e-9, S21dd_denoised_cut, label = "real data")
            plt.xlabel("$Frequency(GHz)$")
            plt.xticks(fontsize=25)
            plt.ylabel("Intensity")
            # fig.show()
            
            if save_name != None:
            
                plt.savefig("denoised_" + save_name)
    
    def denoising(self, fnoise=20):
        """
        Denoises the S21 derivative divide data.

        :param S21dd_cut: the fieldcut segment that is to be denoised
        :param fnoise: dictates the "intensity" of the denoising. Smaller values result in stronger
        denoising, but also in more distorted data

        :return: self.Freq.cut - the 1D array containing the frequencies for a given cut. It was added here for convinience when this function is called outside of the class to just get the plottable fit data,
        S21dd_denoised - the denoised fieldcut segment
        """
        
        Zfft = fft(self.S21dd_cut)
        sample_freq = fftfreq(self.S21dd_cut.size, d=0.001)
        high_freq_fft = Zfft.copy()
        high_freq_fft[np.where(fnoise<np.abs(sample_freq))]=0
        S21dd_denoised = ifft(high_freq_fft)
        
        return self.Freq_cut, S21dd_denoised
   
    def one_peak(self, resonance = [2.2e9], params=None, details=False):
        """
        Initializes the parameters and the model for a one peak fit.

        :param resonance: the expected resonance frequency of the first peak
        :param params: (optional) input of initial parameters for the fit. If it is nonempty the resonance frequency considered for the fit will be taken from here rather than from :param resonance:
        :details: if True prints out the fit information including the error data and possible correlations between parameters

        :return: self.Freq.cut - the 1D array containing the frequencies for a given cut. It was added here for convinience when this function is called outside of the class to just get the plottable fit data,
        our_fit - the fit data of the fieldcut of the spectrum,
        fit_params - fit parameters in the following order: A1, Psi1, fres1, Df1, mod1, Msat_1, H0_1
        """  
        if params == None:
            
            if self.Msat == None:
                
                Msat = 5

            else:
            
                Msat = self.Msat

            parameters = np.array([ 5.46470576e+02, #A
                        -0.97045439,#-0.97045439, #Psi
                        resonance[0],#2.2e+09,  #fres
                        0.25e+09,   #Df
                        1e-3, #mod  
                        Msat, #Msat
                        self.H0 #H0 
                                ])
        else:
            parameters = params
        
        if self.derivative_divide:

            Smod = Model(dS21, prefix="m1_")
        
        else:
            Smod = Model(Lorentzian, prefix='m1_')

        params0 = Smod.make_params()
        params0["m1_A"].set(value= parameters[0])
        params0["m1_Psi"].set(value= parameters[1])
        params0["m1_fres"].set(value= parameters[2])#, max=2.6e9)
        params0["m1_Df"].set(value= parameters[3])
        params0["m1_mod"].set(value= parameters[4], vary=False)
        if self.Msat == None:

            params0["m1_Msat"].set(value= parameters[5], min=0)

        else:

            params0["m1_Msat"].set(value= parameters[5], vary=False)
        
        params0["m1_H0"].set(value= parameters[6], vary=False)
        
        our_fit, fit_params, report = self.fit(model=Smod, params_init=params0, details=details)

        return self.Freq_cut, our_fit, fit_params,report
    
    def two_peak(self, resonance=(2.2e9, 2.2e9), params=None, details=False):
        """
        Initializes the parameters and the model for a two peak fit.

        :param resonance: the expected resonance frequency frequencies (tuple) of the first two peaks
        :param params: (optional) input of initial parameters for the fit. If it is nonempty the resonance frequency considered for the fit will be taken from here rather than from :param resonance:
        :details: if True prints out the fit information including the error data and possible correlations between parameters

        :return: self.Freq.cut - the 1D array containing the frequencies for a given cut. It was added here for convinience when this function is called outside of the class to just get the plottable fit data,
        our_fit - the fit data of the fieldcut of the spectrum,
        fit_params - fit parameters in the following order: A1, Psi1, fres1, Df1, mod1, Msat_1, H0_1,
        A2, Psi2, fres2, Df2, mod2, Msat_2
        """  

        if params == None:

            if self.Msat == None:
                
                Msat = 5

            else:
            
                Msat = self.Msat

            parameters = np.array([5.46470576e+02,    #A
                        -1.97045439, #Psi
                        resonance[0],#2.35e+09,  #fres
                        1e+07,   #Df
                        1e-3, #mod 
                        Msat, #Msat_1
                            self.H0, #H0_1
                        2.99458785e+03,    #A
                        -0.1, #Psi
                        resonance[1],#2.55e+09,  #fres
                        1e7,   #Df
                        1e-3, #mod  
                        Msat, #Msat_2
                            self.H0 #H0_2    
                        ]) 

        else:

            parameters = params

        if self.derivative_divide:
            
            Smod = Model(dS21, prefix="m1_") + Model(dS21, prefix="m2_") 
        
        else:

            Smod = Model(Lorentzian, prefix="m1_") + Model(Lorentzian, prefix="m2_")

        params0 = Smod.make_params()
        params0["m1_A"].set(value= parameters[0])
        params0["m1_Psi"].set(value= parameters[1])
        params0["m1_fres"].set(value= parameters[2])#,max=2.55e9)
        params0["m1_Df"].set(value= parameters[3])
        params0["m1_mod"].set(value= parameters[4], vary=False)

        if self.Msat == None:

            params0["m1_Msat"].set(value= parameters[5], min=0)

        else:

            params0["m1_Msat"].set(value= parameters[5], vary=False)

        params0["m1_H0"].set(value= parameters[6], vary=False)
        params0["m2_A"].set(value= parameters[7])
        params0["m2_Psi"].set(value= parameters[8])
        params0["m2_fres"].set(value= parameters[9])
        params0["m2_Df"].set(value= parameters[10])
        params0["m2_mod"].set(value= parameters[11], vary=False)

        if self.Msat == None:

            params0["m2_Msat"].set(value= parameters[12], min=0)

        else:

            params0["m2_Msat"].set(value= parameters[12], vary=False)

        params0["m2_Msat"].set(value= parameters[12], vary=False)
        params0["m2_H0"].set(value= parameters[13], vary=False)
        
        our_fit, fit_params,report = self.fit(model=Smod, params_init=params0, details=details)

        return self.Freq_cut, our_fit, fit_params

    def three_peak(self, resonance=(2.2e9, 2.2e9, 2.2e9), params=None, details=False):
        """
        Initializes the parameters and the model for a one peak fit. Very unreliable. 
        For a sucessful fit the parameters array defined at the begininng of this function probably needs to be tuned to be closer to the expected values of the fit.

        :param resonance: the expected resonance frequency frequencies (tuple) of the first three peaks
        :param params: (optional) input of initial parameters for the fit. If it is nonempty the resonance frequency considered for the fit will be taken from here rather than from :param resonance:
        :details: if True prints out the fit information including the error data and possible correlations between parameters

        :return: self.Freq.cut - the 1D array containing the frequencies for a given cut. It was added here for convinience when this function is called outside of the class to just get the plottable fit data,
        our_fit - the fit data of the fieldcut of the spectrum,
        fit_params - fit parameters in the following order: A1, Psi1, fres1, Df1, mod1, Msat_1, H0_1,
        A2, Psi2, fres2, Df2, mod2, Msat_2, H0_2, A3, Psi3, fres3, Df3, mod3, Msat_3, H0_3 
        """  

        if params == None:
            
            if self.Msat == None:
                
                Msat = 5

            else:
            
                Msat = self.Msat

            parameters = np.array([5.46470576e+02,    #A
                        -1.97045439, #Psi
                        resonance[0],  #fres
                        1e+07,   #Df
                        2e-3, #mod
                        Msat, #Msat_1
                            self.H0, #H0_1  
                        2.99458785e+03,    #A
                        -0.1, #Psi
                        resonance[1],  #fres
                        1e7,   #Df
                        2e-3, #mod
                        Msat, #Msat_2
                            self.H0, #H0_2 
                        2.99458785e+03,    #A
                        -0.1, #Psi
                        resonance[2],  #fres
                        1e7,   #Df
                        2e-3, #mod
                        Msat, #Msat_3
                            self.H0, #H0_3     
                        ])  

        else:

            parameters = params

        if self.derivative_divide:
            
            Smod = Model(dS21, prefix="m1_") + Model(dS21, prefix="m2_") + Model(dS21, prefix="m3_") 

        else:
            
            Smod = Model(Lorentzian, prefix="m1_") + Model(Lorentzian, prefix="m1_") + Model(Lorentzian, prefix="m1_")

        params0 = Smod.make_params()
        params0["m1_A"].set(value= parameters[0])
        params0["m1_Psi"].set(value= parameters[1])
        params0["m1_fres"].set(value= parameters[2],min=2e9 ,max=3e9)
        params0["m1_Df"].set(value= parameters[3])
        params0["m1_mod"].set(value= parameters[4], vary=False)
        
        if self.Msat == None:

            params0["m1_Msat"].set(value= parameters[5], min=0)

        else:

            params0["m1_Msat"].set(value= parameters[5], vary=False)

        params0["m1_H0"].set(value= parameters[6], vary=False)
        params0["m2_A"].set(value= parameters[7])
        params0["m2_Psi"].set(value= parameters[8])
        params0["m2_fres"].set(value= parameters[9],min=2e9 ,max=5e9)
        params0["m2_Df"].set(value= parameters[10])
        params0["m2_mod"].set(value= parameters[11], vary=False)
        
        if self.Msat == None:

            params0["m2_Msat"].set(value= parameters[12], min=0)

        else:

            params0["m2_Msat"].set(value= parameters[12], vary=False)

        params0["m2_H0"].set(value= parameters[13], vary=False)
        params0["m3_A"].set(value= parameters[14])
        params0["m3_Psi"].set(value= parameters[15])
        params0["m3_fres"].set(value= parameters[16],min=2e9 )
        params0["m3_Df"].set(value= parameters[17])
        params0["m3_mod"].set(value= parameters[18], vary=False)

        if self.Msat == None:

            params0["m3_Msat"].set(value= parameters[19], min=0)

        else:

            params0["m3_Msat"].set(value= parameters[19], vary=False)

        params0["m3_H0"].set(value= parameters[20], vary=False)

        our_fit, fit_params, report = self.fit(model=Smod, params_init=params0, details=details)
        
        return self.Freq_cut, our_fit, fit_params

    def fit(self, model, params_init, details):
        """
        Function that fits our linecut for a given model and initial parameters.

        :param model: a lmfit Model object representing the model we fit our function to
        :param params_init: initial parameters for our fit
        :details: a boolean variable. If True the details of the fit will be printed out (errors and correlations)

        :return: our_fit - the fit data of the fieldcut of the spectrum,
        fit_params - fit parameters corresponding to our model
        """
        
        __, S21dd_denoised = self.denoising()
        out0 = model.fit(S21dd_denoised, params_init, x = self.Freq_cut)

        our_fit = np.real(out0.best_fit[:])
        
        fit_params = []
        pnames = list(out0.params)
        
        for i in range(len(pnames)):
            
            fit_params.append(out0.params[pnames[i]].value)

        if details:
            
            print(out0.fit_report(show_correl=True))
        report=out0
        return our_fit, fit_params,report