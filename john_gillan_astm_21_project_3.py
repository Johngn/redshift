import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare
import statistics

# load data
test_data = np.loadtxt(
    'http://www.astro.lu.se/Education/utb/ASTM21/P3data/P3data02.txt', skiprows=3)

f = test_data[:, 0]             # frequencies
lam = 10 ** test_data[:, 1]     # convert from log to wavelenght
w = test_data[:, 2]             # uncertainties

# plot entire spectrum
plt.figure(figsize=(10, 4))
plt.axvline(x=6115, label='Mg II')
plt.plot(lam, f, linewidth=0.3, color='black')
plt.xlim(np.amin(lam), np.amax(lam))
plt.xlabel(r'Wavelength ($\dot{A}$)')
plt.ylabel(r'Flux ($erg cm^{-2} s^{-1} \dot{A}$)')
plt.grid(which='both')
plt.title('Galaxy Spectrum')
plt.legend()
plt.savefig('spectrum', bbox_inches='tight')

mask = np.logical_and(lam > 5600, lam < 6600)   # area of spectrum needed

f = f[mask]
w = w[mask]
lam = lam[mask]

# plot spectrum of zoomed in area
plt.figure(figsize=(10, 4))
plt.plot(lam, f, linewidth=0.5, color='black')
plt.xlim(5600, 6600)
plt.xticks(ticks=np.arange(5600, 6600, 100))
plt.xlabel(r'Wavelength ($\dot{A}$)')
plt.ylabel(r'Flux ($erg cm^{-2} s^{-1} \dot{A}$)')
plt.grid(which='both')
plt.title('Area of the spectrum under investigation')
plt.savefig('spectrum_zoom', bbox_inches='tight')


def gaussian(x):
    ''' returns gaussian line profile '''
    return np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi)


def lorentzian(x):
    ''' returns lorentzian line profile '''
    return 1 / (np.pi * (1 + x ** 2))


def f_lam_theta(x):
    ''' returns f(lam, theta) using either gaussian or lorentzian '''
    return x[0] + x[1] * (lam - lam_ref) + x[2] * lorentzian((lam - x[3]) / x[4])


def chi_2(x):
    ''' returns minimization function '''
    return np.sum((f - f_lam_theta(x)) ** 2 * w)


# inital guesses for theta
theta = [10, 0, 8, 6115, 70]
lam_ref = 6115

# methods from scipy.optimize.minimize
minimize_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                    'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']

# call minimize function using all methods to get array of theta hat values
theta_hat_results = [minimize(
    chi_2, theta, method=minimize_methods[i], tol=1e-6).x for i in range(len(minimize_methods))]

# create fits of the data using returned theta hat values
fitted_values = [f_lam_theta(theta_hat_results[i])
                 for i in range(len(theta_hat_results))]

# calculate redshift for all theta hat 4 values
z_hat = [(theta_hat_results[i][3] - 2800.3) /
         2800.3 for i in range(len(minimize_methods))]

# calculate residuals
residuals = [f - fitted_values[i] for i in range(len(fitted_values))]

# calculate reduced chi square using all fitted values
reduced_chi2 = [np.sum((f - fitted_values[i]) ** 2 * w) / (len(f) - 5)
                for i in range(len(minimize_methods))]


def chi_2_2(x):
    ''' returns minimization function for synthetic data '''
    return np.sum((f_synt - f_lam_theta(x)) ** 2 * w)


# initialize array for redshifts of synthetic data
z_hats = []

# calculate redshift for 1000 different synthetic data sets
for i in range(100):
    # generate random synthetic data set
    f_synt = fitted_values + (1 / np.sqrt(w)) * np.random.normal(0, 1, 1)

    # call minimization function using only Nelder-Mead
    theta_hat = minimize(chi_2_2, theta, method='Nelder-Mead', tol=1e-6).x

    # calculate redshift for synthetic data set
    z_hat = (theta_hat[3] - 2800.3) / 2800.3

    # append to array of  redshifts
    z_hats.append(z_hat)

# calculate standard deviation of redshifts of synthetic data sets
sampleSD = statistics.stdev(z_hats)

# plot fitted lines for all methods of minimization
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.subplots_adjust(hspace=.5, wspace=.4)
axs = axs.ravel()

for i in range(9):
    axs[i].plot(lam, fitted_values[i], color='r')
    axs[i].plot(lam, f, linewidth=0.3, color='black')
    axs[i].set_xlim(5600, 6600)
    axs[i].set_xticks(ticks=np.arange(5600, 6600, 200))
    axs[i].set_xlabel(r'Wavelength ($\dot{A}$)')
    axs[i].set_ylabel(r'Flux ($erg cm^{-2} s^{-1} \dot{A}$)')
    axs[i].grid(which='both')
    axs[i].set_title('Fitted line (' + minimize_methods[i] + ')')

plt.savefig('fitted_gaussian_9', bbox_inches='tight')

# plot residuals for all methods of minimization
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.subplots_adjust(hspace=.5, wspace=.4)
axs = axs.ravel()

for i in range(9):
    axs[i].plot(lam, residuals[i], linewidth=0.3, color='black')
    axs[i].set_xlim(5600, 6600)
    axs[i].set_xticks(ticks=np.arange(5600, 6600, 200))
    axs[i].set_xlabel(r'Wavelength ($\dot{A}$)')
    axs[i].set_ylabel(r'Flux ($erg cm^{-2} s^{-1} \dot{A}$)')
    axs[i].grid(which='both')
    axs[i].set_title('Residuals (' + minimize_methods[i] + ')')

plt.savefig('residuals_gaussian_9', bbox_inches='tight')
