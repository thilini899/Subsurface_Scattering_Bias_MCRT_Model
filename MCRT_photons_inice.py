"""
Monte Carlo Radiative Simulation for Photon Transport in Glacier Ice
--------------------------------------------------------------------

Author      : Thilini Bamunu Arachchige ( t.thilakarathne@und.edu )
Affiliation : University of North Dakota, Department of Physics and Astrophysics
Advisor     : Dr. Markus Allgaier
Description :
    This code simulates photon propagation in glacier ice using
    Monte Carlo radiative transfer techniques. This contains the 
    time of flight histograms of Incident and Backscattered pulses 
    with the corresponding medians of the histograms and its uncertainties.

The model includes :
      - Henyey–Greenstein scattering
      - Beer–Lambert absorption
      - Fresnel reflection and total internal reflection
      - Time-of-flight histogram analysis
      - Bootstrap-based uncertainty estimation

Applications :
    - ICESat-2 ATL03 / ATL06 subsurface scattering bias
    - Light penetration depth estimation
    - Instrument impulse-response modeling

Date of Published:
"""
import csv
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from scipy.interpolate import interp1d

start = time.time()         # Start point to measure the execution time

num_particles = 25       # Number of particles enter the medium
c = 3e8                     # Speed of light in vacuum (m/s)
n_ice = 1.33                # Refractive index in ice
speed_in_ice = c / n_ice    
# FWHM = 0.5e-9               # Full width half maximun of the incoming laser pulse 
# sigma = FWHM / 2.35
scattering_length = 0.1  # meters, Scattering length in ice
absorption_length = 100     # meters, Absorption length in ice
angle_deg = 0.0            # degrees, Angle of Incident
num_bootstrap=500
confidence=0.68

theta0 = np.radians(angle_deg)
init_dir = np.array([np.sin(theta0), 0.0, 0.0])
beam_radius = 6 # meters

def fresnel_reflectance(n1, n2, theta_i):
    """
    Compute Fresnel reflectance for an unpolarized photon
    incident from medium n1 into medium n2 at incident angle theta_i
    (theta_i measured from the surface normal).
    
    
    If the incident angle is >= 90 degrees, the photon is grazing or
    propagating parallel to the surface; treat as total reflection.
    """
    
    if theta_i >= np.pi / 2:
        return 1
    sin_t2 = (n1 / n2) * np.sin(theta_i)

    #If sin(theta_t) > 1, refraction is not possible and total internal reflection occurs.
    if sin_t2 > 1:
        return 1

    #Compute the transmitted (refracted) angle theta_t.
    theta_t = np.arcsin(np.clip(sin_t2, -1, 1))                 #np.clip is used to protect against small numerical overshoots that could cause invalid values in arcsin.

    #Fresnel reflection coefficient for s-polarized light
    #(electric field perpendicular to the plane of incidence)
    rs = (
        (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) /
        (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
    ) ** 2

    #Fresnel reflection coefficient for p-polarized light
    #(electric field parallel to the plane of incidence)
    rp = (
        (n1 * np.cos(theta_t) - n2 * np.cos(theta_i)) /
        (n1 * np.cos(theta_t) + n2 * np.cos(theta_i))
    ) ** 2

    #Return the reflectance for unpolarized light, obtained by averaging s and p polarization components.
    return 0.5 * (rs + rp)


def total_internal_reflection(n1, n2, theta_i):
    """
    Determine whether total internal reflection (TIR) occurs
    for a photon incident from medium n1 into medium n2.
    
    """

    #The critical angle condition is: sin(theta_i) > n2 / n1

    #If both conditions are satisfied, refraction is forbidden and the photon is fully reflected at the interface.
    return (n1 > n2) and (np.sin(theta_i) > (n2 / n1))


def henyey_greenstein_sample(g):
    """
    Sample a scattering polar angle theta from the Henyey–Greenstein (HG) phase function using
    inverse transform sampling.
    
    g --> asymmetry parameters:      g >  0   → forward-peaked scattering
    """
    e = random.uniform(0, 1)            #Uniform random number in [0, 1] used to sample the cumulative distribution function (CDF)

    theta = np.arccos(
        (1 + g**2 - ((1 - g**2) / (1 + g - 2 * g * e))**2) / (2 * g)
    )

    #Return the sampled polar scattering angle (radians)
    return theta


def rodrigues_rotation(v, k, theta):
    """
    Rotate a 3D vector v by an angle theta about an axis k
    using Rodrigues' rotation formula.
    
    v     : input vector to be rotated (e.g., photon direction)
    k     : rotation axis (does not need to be unit length)
    theta : rotation angle in radians (right-hand rule)
    """
    
    #Convert the rotation axis to a NumPy array and normalize it to ensure a proper unit rotation axis.
    k = np.asarray(k) / np.linalg.norm(k)

    #Rodrigues' rotation formula:
    #v_rot = v*cos(theta)+ (k × v)*sin(theta)+ k*(k · v)*(1 - cos(theta))
    #Term 1: Component of v preserved along its original direction
    #Term 2: Component perpendicular to both k and v (causes rotation)
    #Term 3: Component of v parallel to the rotation axis k
    #The rotation follows the right-hand rule about axis k.
    return (
        v * np.cos(theta) +
        np.cross(k, v) * np.sin(theta) +
        k * np.dot(k, v) * (1 - np.cos(theta))
    )

def unitvector_conversion(vector):
    """
    Convert an arbitrary vector into a unit (normalized) vector.
    """
    return vector / np.linalg.norm(vector)

def check_photon_escape(s, z, photon_is_absorbed, Time, j):
    """
    Determine whether a photon escapes the ice, is reflected at the surface,
    or is absorbed within the medium.
    
    s                   : total path length traveled since last interaction
    z                   : vertical photon position relative to the surface
                          (z >= 0 indicates the photon has reached the surface)
    photon_is_absorbed  : flag for absorption state (not updated here)
    Time                : photon time-of-flight at this step
    j                   : photon index for storing time-of-flight
    
    """
    #Probability that the photon survives absorption over path length s according to Beer–Lambert law: P_survival = exp(-s / absorption_length)
    P_abs = np.exp(-1 * (s / absorption_length))

    #Case 1: Photon has reached or crossed the ice–air interface
    if z >= 0:

        #Compute the incident angle at the surface relative to the normal.
        theta_i = np.arcsin(min(1, abs(z) / s))

        #Check for total internal reflection (ice → air interface)
        #If TIR occurs, the photon is reflected back into the ice.
        if total_internal_reflection(n_ice, 1.0, theta_i):
            return 0

        #If not TIR, compute Fresnel reflectance.
        if random.uniform(0, 1) < fresnel_reflectance(n_ice, 1.0, theta_i):
            return 0

        #Photon successfully transmits through the surface and escapes. store its time-of-flight
        time_of_flight[j] = Time
        return 1

    #Case 2: Photon is still inside the ice volume (z < 0)
    #Perform an absorption test using Beer–Lambert statistics.
    if random.uniform(0, 1) > P_abs:
        #Photon is absorbed within the medium
        return 1

    #Photon neither escaped nor was absorbed in this step
    return 0

def bootstrap_median(data, num_bootstrap, confidence):
    """
    Estimate the median of a dataset and its uncertainty using bootstrap resampling.
    """
    medians = []                #List to store the median from each bootstrap resample
    n = len(data)               #Number of data points in the original sample

    #Perform bootstrap resampling
    for _ in range(num_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(resample))

    #Compute lower and upper bounds of the confidence interval from the empirical distribution of bootstrap medians.
    #For confidence = 0.68, this corresponds to the 16th and 84th percentiles.
    lower = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper = np.percentile(medians, (1 + confidence) / 2 * 100)

    #Compute the median of the original (non-resampled) dataset
    median = np.median(data)

    #Return:
    #median : point estimate of the median
    #lower  : lower bound of the confidence interval
    #upper  : upper bound of the confidence interval
    #upper - lower : total width of the confidence interval
    return median, lower, upper, upper - lower

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

num_particles=len(accepted_x)       # Number of photons inserts the medium is equal to the length of accepted_x values
print( "Number of photons goes into ice =",num_particles)

x0 = np.random.normal(0, beam_radius / 2, num_particles)
y0 = np.random.normal(0, beam_radius / 2, num_particles)
time_of_flight = np.zeros(num_particles)

entry_times = T0 + (x0 * np.tan(theta0)) / c                                    #Intial timings of photons

#Rotate the axis about y axis
rotation_matrix = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

#Project the propagation of photons in the medium.    
#Convention used here:
#photon_is_absorbed == 0  -> photon is still "alive" (continue propagating)
#photon_is_absorbed == 1  -> photon terminated (either absorbed in volume or escaped at surface)
for j in range(num_particles):
    P = [x0[j], y0[j], 0]                                                   # Initial position of the photon                        
    D = init_dir.copy()
    photon_is_absorbed = 0                                                 
    s = -scattering_length * np.log(1 - random.uniform(0, 1)) * (1 - 0.75)  # total path length traveled since last interaction
    Time = entry_times[j]                                                   # Update time the photon
    D[2] += -s                                                              # vertical photon position relative to the surface
    P[2] += D[2]                                                            # Update position of photon
    Time += np.sqrt(D[0]*D[0] + D[1]*D[1] + D[2]*D[2])/ speed_in_ice        # Update time in seconds
    v = D[:]
    k_rot = np.dot(rotation_matrix, v)                                      # Rotate around the y axis

    #Propagate a single photon through the scattering medium until it is terminated.
    while photon_is_absorbed == 0:
    
        #1) Sample the polar scattering angle from Henyey–Greenstein (HG) ---
        #g = 0.75 gives forward-peaked scattering.
        theta1 = henyey_greenstein_sample(0.75)
    
        #2) Apply the polar deflection about an axis perpendicular to current direction
        #Rodrigues rotation rotates the current direction v by theta1 around k_rot.
        #k_rot is perpendicular to v, acting as the "rotation axis" that defines the scattering plane for the polar angle.
        new_direction1 = rodrigues_rotation(v, k_rot, theta1)
    
        #3) Normalize and scale by current step length s 
        new_direction1_unit = s * unitvector_conversion(new_direction1)
    
        #4) Sample a random azimuthal angle phi uniformly in [0, 2π) 
        #This randomizes the scattering direction around the original propagation axis.
        theta2 = random.uniform(0, 2 * math.pi)
    
        new_v = new_direction1_unit
    
        #5) Sample the next free path length between scattering events 
        s = -scattering_length * np.log(1 - random.uniform(0, 1)) * (1 - 0.75)
    
        #6) Apply azimuthal rotation around the current axis
        #Rotate new_v by theta2 around axis v.
        #This step is intended to randomize the azimuth while keeping the polar angle fixed.
        new_direction2 = rodrigues_rotation(new_v, v, theta2)
    
        #7) Convert the new direction into a displacement step of length s 
        new_direction2_unit = s * unitvector_conversion(new_direction2)
    
        #Store the current displacement in D (in-place update).
        D[:] = new_direction2_unit
    
        # 8) Update the photon position P by adding the step displacement
        P[0] += D[0]
        P[1] += D[1]
        P[2] += D[2]
    
        #9) Update time-of-flight
        Time += np.sqrt(D[0]*D[0] + D[1]*D[1] + D[2]*D[2]) / speed_in_ice
    
        #10) Termination check: surface escape or absorption
        #if the photon reaches the surface: applies TIR + Fresnel reflection; if transmitted, records ToF
        #if still in volume: applies Beer–Lambert absorption test
        photon_is_absorbed = check_photon_escape(s, D[2], photon_is_absorbed, Time, j)
    
        #11) Update the photon direction state for the next iteration 
        #v becomes the new (scaled) direction/displacement vector used next step.
        v = new_direction2_unit
    
        #12) Update k_rot for the next polar-angle rotation 
        #rotation_matrix here is being used to generate an axis orthogonal to v
        #This provides a "perpendicular" axis for the next Rodrigues rotation.
        k_rot = np.dot(rotation_matrix, v)

                
#Get medians and uncertainties
med1, low1, high1, unc1 = bootstrap_median(time_of_flight,num_bootstrap, confidence)
med2, low2, high2, unc2 = bootstrap_median(entry_times, num_bootstrap, confidence)
   
print(f"Escaped photons: {np.count_nonzero(time_of_flight)}")

edges = np.linspace(-2.5e-9,8e-9,100)
fig, ax = plt.subplots()

#1) Plot the "incident/entry" timing distribution (ATLAS instrument response)
x = med2                                                    #x is the central estimate (median) of the entry-time distribution
counts, bin_edges = np.histogram(entry_times, bins=edges)   #counts: number of samples per bin,bin_edges: the edges actually used 
max_count1 = np.max(counts)                                 #Peak bin count (maximum y-value) for incident histogram.
y = max_count1/2                                            #y-position to place the horizontal error bar.
plt.hist(entry_times, bins=edges, alpha=0.8, label='ATLAS response')                    #Plot the entry-time histogram.
plt.errorbar(x, y, xerr=unc2, color='black', capsize=4, elinewidth=1, markersize=4)     #Plot a horizontal error bar centered at x=med2 with half-width unc2.

#2) Plot the backscattered time-of-flight distribution
x=med1                                                      #x is the central estimate (median) of the backscattered distribution
counts, bin_edges = np.histogram(time_of_flight, bins=edges)#Compute histogram counts for time_of_flight using the same bin edges.
max_count = np.max(counts)                                  #Peak bin count for the backscattered histogram
y=max_count/2                                               #Place the backscattered median error bar at half of this peak height
plt.hist(time_of_flight, bins=edges, alpha=0.7, label='Backscattered Pulse', color='darksalmon')    #Plot the time-of-flight histogram.
plt.errorbar(x, y, xerr=unc1, color='black', capsize=4, elinewidth=1, markersize=4)                 #Plot a horizontal error bar for the backscattered median.

#3) Estimate "penetration depth" from the histogram peak location
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2          #Compute bin centers from bin_edges for the TOF histogram.
max_x_value = bin_centers[np.argmax(counts)]                #Find the x-value (time) corresponding to the maximum count bin.
light_penetration = (3e8 * max_x_value * 2) / (2 * 1.33)    #Convert time delay to penetration depth estimate.
print(f"x value at Max_y : {max_x_value}")
print(f"Light Penetration Depth= {light_penetration}")
plt.axvline(x=med2, color='blue', linestyle='--', linewidth=1.5)        #Plot a dashed vertical line at the Incident median
plt.axvline(x=med1, color='orangered', linestyle='--', linewidth=1.5)   #Plot a dashed vertical line at the backscattered median
plt.xlim(-3e-9,10e-9)
plt.xlabel('Time of flights [s]')
plt.ylabel('Detector Counts')
plt.legend(['Incident Pulse','Backscattered Pulse'],loc='upper right',)
#plt.title(f"Angle of Incident = {angle_deg} degrees, Sca.Length={scattering_length} m,\nNumber of Photons = {num_particles}" )
plt.tight_layout()
plt.savefig(f'TOF_plot{angle_deg , scattering_length , num_particles}.svg', format='svg', dpi=1200)
plt.show()

end = time.time()            #End point to measure the execution time
print(f"\nTime taken for simulation: {end - start:.2f} s")
