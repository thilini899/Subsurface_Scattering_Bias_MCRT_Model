"""
Author      : Thilini Bamunu Arachchige ( t.thilakarathne@und.edu )
Affiliation : University of North Dakota, Department of Physics and Astrophysics
Advisor     : Dr. Markus Allgaier
Description :
    Smooth plot.
    Scattering bias and error of light penetration were calculated using median. 
    Correction for light penetration depth has included. Heatmap of light penetration depth.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from lightpen_depths import m  # Correction slope from previous fit
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.stats import exponnorm
from scipy.interpolate import interp1d
import csv

start = time.time()

num_particles = 171
c = 3e8                 # Speed of light in vacuum (m/s)
n_ice = 1.33            #refractive index of ice
speed_in_ice = c / n_ice
absorption_length = 100     #absorption length in ice
angles = np.arange(0, 10, 0.5)
scattering_lengths = np.arange(0.01, 0.1, 0.005)    #scattering length in ice
light_penetrationdep = np.zeros((len(scattering_lengths), len(angles)))
bias_penetrationdep= np.zeros((len(scattering_lengths), len(angles)))
dep_values = defaultdict(list)
bias_values = defaultdict(list)

class left_emg_gen:
    """
    Exponentially Modified Gaussian (EMG) with left tail 
    """
    def cdf(self, x, K, loc=0, scale=1):
        return 1 - exponnorm.cdf(-x, K=K, loc=-loc, scale=scale)

    def pdf(self, x, K, loc=0, scale=1):
        return exponnorm.pdf(-x, K=K, loc=-loc, scale=scale)

left_emg = left_emg_gen()

def get_fit_hist(bins, times, sigmas, Ks, pulse_norms):
    """
    Evaluate the functional form for a given binning.
    """
    x = (bins[:-1] + bins[1:]) / 2.0    # Compute bin centers
    bin_widths = bins[1:] - bins[:-1]   # Bin width
    fit_hist = np.zeros_like(x)

    # Sum up EMG components
    for (t, s, k, n) in zip(times[:-1], sigmas[:-1], Ks[:-1], pulse_norms[:-1]):
        fit_hist += n * exponnorm.pdf(x, K=k, loc=t, scale=s) * bin_widths

    # Add left-tailed EMG component
    fit_hist += pulse_norms[-1] * left_emg.pdf(x, Ks[-1], loc=times[-1], scale=sigmas[-1]) * bin_widths
    
    return fit_hist, x

def read_csv_header(filename):
    """
    Reads the header of a CSV file and extracts fit parameters
    """
    header_data = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        row_counter = 0
        for row in reader:
            if len(row) > 1:
                header_data[row[0]] = row[1:]
            row_counter += 1
            if row_counter >= int(header_data.get("Header_Len", ["0"])[0]):  
                break
    return header_data
def rejection_sampling(x,fit_hist_interp,target_smaples):
    """
    Generate observations from ATL06 response function using rejection sampling
    """
    accepted_data = []
    x = np.array(x)
    fit_hist_interp = np.array(fit_hist_interp)
    
    while len(accepted_data) < target_smaples:
        for xi, hist in zip(x, fit_hist_interp):
            yi = np.random.rand()
    
            if (hist > yi) and (yi >= 0):
                accepted_data.append(xi)
            if len(accepted_data) >= target_smaples:
                break
    return np.array(accepted_data)

#Read data from ATLAS impulse response function
filename = "./LinearImpulseResponse_AA2_2024326_A.csv"   #File path for ATLAS response 
header_info = read_csv_header(filename)
ispot = 1
bins = np.linspace(-50, 400, int(450 / 0.05) + 1)
fit_hist, fit_x = get_fit_hist(
    bins,
    np.array(header_info[f"Fit_params_spot{ispot}_times"], dtype=np.float64),
    np.array(header_info[f"Fit_params_spot{ispot}_sigmas"], dtype=np.float64),
    np.array(header_info[f"Fit_params_spot{ispot}_Ks"], dtype=np.float64),
    np.array(header_info[f"Fit_params_spot{ispot}_norms"], dtype=np.float64)
)
fit_hist = fit_hist / np.max(fit_hist)                                          #Normalize the y axis values 
x = np.linspace(-2.5, 5.5, num_particles*50)
random.shuffle(x) 
f_interp = interp1d(fit_x, fit_hist, kind='linear', fill_value="extrapolate")   #Use interpolation to generate y values corresponding to random x values
fit_hist_interp = f_interp(x)
accepted_x = rejection_sampling(x, fit_hist_interp,num_particles)           #After the rejection sampling
T0 = accepted_x *1e-9                                                     #T0 is the intial timing for photons

rotation_matrix = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

def fresnel_reflectance(n1, n2, theta_i):
    if theta_i >= np.pi / 2:
        return 1
    sin_t2 = (n1 / n2) * np.sin(theta_i)
    if sin_t2 > 1:
        return 1
    theta_t = np.arcsin(np.clip(sin_t2, -1, 1))
    rs = ((n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))) ** 2
    rp = ((n1 * np.cos(theta_t) - n2 * np.cos(theta_i)) / (n1 * np.cos(theta_t) + n2 * np.cos(theta_i))) ** 2
    return 0.5 * (rs + rp)

def total_internal_reflection(n1, n2, theta_i):
    return n1 > n2 and np.sin(theta_i) > (n2 / n1)

def henyey_greenstein_sample(g):
    e = random.uniform(0, 1)
    theta = np.arccos((1 + g**2 - ((1 - g**2) / (1 + g - 2 * g * e))**2) / (2 * g))
    return theta

def rodrigues_rotation(v, k, theta):
    k = np.asarray(k) / np.linalg.norm(k)
    return (v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta)))

def unitvector_conversion(vector):
    return vector / np.linalg.norm(vector)

def check_photon_escape(s, z, photon_is_absorbed, Time, j):
    P_abs = np.exp(-1 * (s / absorption_length))
    if z >= 0:
        theta_i = np.arcsin(min(1, abs(z) / s))
        if total_internal_reflection(n_ice, 1.0, theta_i):
            return 0
        if random.uniform(0, 1) < fresnel_reflectance(n_ice, 1.0, theta_i):
            return 0
        time_of_flight[j] = Time
        return 1
    if random.uniform(0, 1) > P_abs:
        return 1
    return 0

def bootstrap_median(data, num_bootstrap=1000, confidence=0.68):
    medians = []
    n = len(data)
    for _ in range(num_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(resample))
    lower = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper = np.percentile(medians, (1 + confidence) / 2 * 100)
    median = np.median(data)
    return median, lower, upper, upper - lower

for l in range(10):
    for j, angle_deg in enumerate(angles):
        theta0 = np.radians(angle_deg)
        init_dir = np.array([np.sin(theta0), 0.0, 0.0])
        x0 = np.random.normal(0, 3.0, num_particles)
        y0 = np.random.normal(0, 3.0, num_particles)
        entry_times = T0 + (x0 * np.tan(theta0)) / c
    
        for i, scattering_length in enumerate(scattering_lengths):
            time_of_flight = np.zeros(num_particles)
    
            for k in range(num_particles):
                P = [x0[k], y0[k], 0]
                D = init_dir.copy()
                photon_is_absorbed = 0
                s = -scattering_length * np.log(1 - random.uniform(0, 1)) * (1 - 0.75)
                Time = entry_times[k]
                D[2] += -s
                P[2] += D[2]
                Time += np.linalg.norm(D) / speed_in_ice
                v = D[:]
                k_rot = np.dot(rotation_matrix, v)
    
                while photon_is_absorbed == 0:
                    theta1 = henyey_greenstein_sample(0.75)
                    new_direction1 = rodrigues_rotation(v, k_rot, theta1)
                    new_direction1_unit = s * unitvector_conversion(new_direction1)
                    theta2 = random.uniform(0, 2 * math.pi)
                    new_v = new_direction1_unit
                    s = -scattering_length * np.log(1 - random.uniform(0, 1)) * (1 - 0.75)
                    new_direction2 = rodrigues_rotation(new_v, v, theta2)
                    new_direction2_unit = s * unitvector_conversion(new_direction2)
                    D[:] = new_direction2_unit
                    P[0] += D[0]
                    P[1] += D[1]
                    P[2] += D[2]
                    Time += np.linalg.norm(D) / speed_in_ice
                    photon_is_absorbed = check_photon_escape(s, D[2], photon_is_absorbed, Time, k)
                    v = new_direction2_unit
                    k_rot = np.dot(rotation_matrix, v)
    
            med1, _, _, unc1 = bootstrap_median(time_of_flight)
            med2, _, _, unc2 = bootstrap_median(entry_times)
            time_difference = med1 - med2
            
            light_penetration = (c * time_difference) / (2 * n_ice)
            light_penetration -= m * angle_deg  # Correction
            light_penetrationdep[i][j] = light_penetration
            
            errorofdepth = (c * np.sqrt(unc1**2 + unc2**2)) / (2 * n_ice)
            bias_penetrationdep[i][j]= errorofdepth
            
            dep_values[(i, j)].append(light_penetrationdep[i][j])
            bias_values[(i, j)].append(bias_penetrationdep[i][j])
            
# average over l for each (j,i)
dep_averages = {key: (sum(val)/len(val) if len(val) > 0 else 0) 
            for key, val in dep_values.items()}
# average over l for each (j,i)
bias_averages = {key: (sum(val)/len(val) if len(val) > 0 else 0) 
            for key, val in bias_values.items()}

# Define the two colors
colors = ['steelblue', 'tomato']  # Example: blue for 0, red for 1

# Create a custom colormap
cmap = ListedColormap(colors)

# Create a 3D RGB array for custom colors
color_map = np.zeros((light_penetrationdep.shape[0],
                      light_penetrationdep.shape[1], 3))

for i in range(light_penetrationdep.shape[0]):
    for j in range(light_penetrationdep.shape[1]):
        if not np.isnan(light_penetrationdep[i, j]):                                            
            if abs(dep_averages[(i, j)]) - bias_averages[(i, j)] > 0:
            #if light_penetrationdep[i, j] - bias_penetrationdep[i, j] > 0:
                color_map[i, j] = plt.get_cmap('summer')(0.6)[:3]  # greenish
            else:
                color_map[i, j] = plt.get_cmap('copper')(0.6)[:3]  # brownish

# Define the legend handles with fixed colors matching your colormap
green_patch = mpatches.Patch(color=plt.get_cmap('summer')(0.6), label='Significant Scattering Bias')
brown_patch = mpatches.Patch(color=plt.get_cmap('copper')(0.6), label='Insignificant Scattering Bias')

fig, ax = plt.subplots()

#plt.figure(figsize=(8, 6))
plt.imshow(color_map, origin='lower', aspect='auto',
            extent=[angles.min(), angles.max(),
                    scattering_lengths.min(), scattering_lengths.max()])
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Scattering Length (m)')
#plt.title(f'Penetration Depth Significance Map\nNumber of Photons = {num_particles}')
plt.legend(handles=[green_patch, brown_patch], loc='upper left')
image_format = 'svg'
image_name = f'penetrationdepth{num_particles}.svg'
fig.savefig(image_name, format=image_format, dpi=1200)
plt.show()
end = time.time()
print(f"\nTime taken for simulation: {end - start:.2f} s")