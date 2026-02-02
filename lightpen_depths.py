"""
Author      : Thilini Bamunu Arachchige ( t.thilakarathne@und.edu )
Affiliation : University of North Dakota, Department of Physics and Astrophysics
Advisor     : Dr. Markus Allgaier
Description :
    This code simulates photon propagation in glacier ice using
    Monte Carlo radiative transfer techniques. This contains Scattering bias 
    and error of light penetration calculated using the medians of populations of
    initial and backscattered pulses. 
    
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from scipy.stats import exponnorm
from scipy.interpolate import interp1d
import csv

start = time.time()
num_particles = 171
c = 3e8  # Speed of light in vacuum (m/s)
n_ice = 1.33
speed_in_ice = c / n_ice
FWHM = 0.5e-9
sigma = FWHM / 2.35
scattering_length = 0.02
absorption_length = 100
num_bootstrap=1000
confidence=0.68

angles=[]
bias_penetrationdep=[]
light_penetrationdep=[]

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

#############################
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

def bootstrap_median(data, num_bootstrap, confidence):
    # data in seconds
    medians = []
    n = len(data)
    
    for _ in range(num_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(resample))
    lower = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper = np.percentile(medians, (1 + confidence) / 2 * 100)
    median = np.median(data)
    
    return median, lower, upper, upper - lower

#############################
for i in np.arange(0, 5, 0.5):    
    time_of_flight = np.zeros(num_particles)
    angle_deg = i
    #print(f"\nAngle of incident : {i:.2f} degrees")
    angles.append(angle_deg)
    theta0 = np.radians(angle_deg)
    init_dir = np.array([np.sin(theta0), 0.0, 0.0])
    beam_radius = 6 # meters
    x0 = np.random.normal(0, beam_radius / 2, num_particles)
    entry_times = T0 + (x0 * np.tan(theta0)) / c
    
    for j in range(num_particles):
        #P = [x0[j], y0[j], 0]
        P = [x0[j],0.0, 0.0]
        D = init_dir.copy()
        photon_is_absorbed = 0
        s = -scattering_length * np.log(1 - random.uniform(0, 1)) * (1 - 0.75)
        Time = entry_times[j]
        D[2] += -s
        P[2] += D[2]
        Time += np.sqrt(D[0]*D[0] + D[1]*D[1] + D[2]*D[2])/ speed_in_ice  # Update time in seconds
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
            Time +=  np.sqrt(D[0]*D[0] + D[1]*D[1] + D[2]*D[2]) / speed_in_ice
            photon_is_absorbed = check_photon_escape(s, D[2], photon_is_absorbed, Time, j)
            v = new_direction2_unit
            k_rot = np.dot(rotation_matrix, v)
                        
    #Get medians and uncertainties
    med1, low1, high1, unc1 = bootstrap_median(time_of_flight,num_bootstrap, confidence)
    med2, low2, high2, unc2 = bootstrap_median(entry_times, num_bootstrap, confidence)
    time_difference = med1 - med2 
    light_penetration = (3e8 * time_difference ) /2.66 #(2 * 1.33)
    light_penetrationdep.append(light_penetration)
    errorofdepth= c* np.sqrt((unc1**2) + (unc2**2))/ 2.66 # (2*1.33)
    bias_penetrationdep.append(errorofdepth)
    
    ############################
    edges = np.linspace(-2.5e-9, 5.5e-9,100)
    fig, ax = plt.subplots()
    plt.hist(T0, bins=edges, alpha=0.8, label='Incident Pulse', edgecolor='black',color='pink')
    plt.hist(entry_times, bins=edges, alpha=0.8, label='Incident Pulse', edgecolor='blue')
    plt.hist(time_of_flight, bins=edges, alpha=0.8, label='Backscattered Pulse', color='darksalmon', edgecolor='darksalmon')
    plt.axvline(x=med1, color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(x=med2, color='red', linestyle='--', linewidth=1.5)
    #plt.ylim(0,1000)
    plt.xlabel(r'$\tau$ [s]')
    plt.ylabel('Counts')
    plt.legend(['Incident Pulse', 'Backscattered Pulse'])
    #plt.legend(['Incident Pulse', f'Backscattered Pulse\nLight Penetration = {light_penetration:.4f} m  +/- {errorofdepth:.3f} m'],fontsize=7, frameon=False)
    plt.title(f"Time Distribution (angle={i:.2f}) (Scattering length={scattering_length})")
    plt.tight_layout()
    plt.savefig(f'TOF_plot{i:.1f}.svg', format='svg', dpi=1200)
    plt.show()
  
    # print(f"Angle {i} -> Light Penetration Depth=   {light_penetration:.3f} m +/- {errorofdepth:.3f} m" )

end = time.time()
print(f"\nTime taken for simulation: {end - start:.2f} s")

##find line of best fit
m, b = np.polyfit(angles, light_penetrationdep, 1)
fit_line_y = np.polyval([m, b], angles)

# plt.subplot(2, 1, 1)
plt.errorbar(angles, light_penetrationdep,yerr=bias_penetrationdep, fmt='o',color='blue' ,ecolor='red', capsize=5, label='Data Without Correction')
#plt.plot(angles,light_penetrationdep,color='purple')
plt.plot(angles, fit_line_y, color='black', linestyle='-',label='Fit Line')
plt.xlabel('Angle of incident( degrees)')
#plt.title(f'Angle vs Light Peneration Depth\n (Sca.Length = {scattering_length} m, Num. of particles ={num_particles}) ')
plt.ylabel('Light Penetration dep. (m)',fontsize=10)
plt.legend()
plt.ylim(-0.05, 0.1)
plt.axhline(0, color='black', linestyle='--', linewidth=1) #plt.title(f'Angle vs Light Peneration (Sca.Length = {scattering_length} m) ')
plt.savefig(f'penetrationwithoutcorrection_{scattering_length}.svg', format='svg', dpi=1200)
plt.show()

light_penetrationdep -= m * np.array(angles)

diff = np.array(light_penetrationdep) - np.array(bias_penetrationdep)

##find line of best fit
m2, b2 = np.polyfit(angles, light_penetrationdep, 1)
fit_line_y2 = np.polyval([m2, b2], angles)

# plt.subplot(2, 1, 1)
fig, ax = plt.subplots() 
# plot data with error bars
ax.errorbar(
    angles,
    light_penetrationdep,
    yerr=bias_penetrationdep,
    fmt='o',
    color='blue',
    ecolor='red',
    capsize=5,
    label='Data with Correction'
)

# fitted line
ax.plot(
    angles,
    fit_line_y2,
    color='black',
    linestyle='-',
    #label='Fit Line'
)

# # set limits before fill_between so ax.get_ylim() is stable
ax.set_ylim(min(light_penetrationdep) - 0.1, max(light_penetrationdep) + 0.1)
y_bottom, y_top = ax.get_ylim()

# Extend x-range slightly to cover the full width
x_extended = np.linspace(angles[0] - 0.1, angles[-1] + 0.1, 500)
diff_interp = np.interp(x_extended, angles, diff)  # interpolate diff

mask = diff_interp > 0
x_pos = x_extended[mask]
y_pos = diff_interp[mask]
max_x = np.max(x_pos)
plt.axvline(max_x, linestyle='--', linewidth=1)

# Fill green where diff > 0
ax.fill_between(
    x_extended, y_bottom, y_top,
    where=(diff_interp > 0),
    facecolor='lightgreen', alpha=0.8,
    interpolate=True, zorder=0, label='Penetration Significant'
)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Angle of incidence (degrees)')
ax.set_ylabel('Light Penetration Depth (m)', fontsize=10)
# ax.set_title(
#     f'Angle vs Light Penetration Depth\n'
#   f'(Sca. Length = {scattering_length} m, Num. of particles = {num_particles})'
# )
ax.set_ylim(-0.05, 0.1)
ax.legend()
#fig.savefig(f'penetrationwithcorrection_{scattering_length}.svg', format='svg', dpi=1200)
plt.show()
print(m2)
