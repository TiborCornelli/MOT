import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from image_utils import read_gz_file, open_all_ims_in_dir


def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


roi = {
    "xMain": np.array([620, 1400]),
    "yMain": np.array([416, 800]),
}

directory_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "AbsorptionData", "2"
)
file_list = open_all_ims_in_dir(directory_path)

x_length = roi["xMain"][1] - roi["xMain"][0]
y_length = roi["yMain"][1] - roi["yMain"][0]

x_vals_centered = np.arange(x_length) - x_length / 2
y_vals_centered = np.arange(y_length) - y_length / 2

file_indices = []
x0_values = []
sigma_x_values = []
y0_values = []
sigma_y_values = []
dt = 250 * 1e-6  # time step in seconds
start_tof = 71 * 1e-6
time_stamps = np.arange(len(file_list)) * dt

# ------- Fitting Gaussian to projections -------

for file in file_list:
    img = read_gz_file(file, returnRaw=False, returnDict=False)
    cropped = img[slice(*roi["yMain"]), slice(*roi["xMain"])]
    projection_x = np.sum(cropped, axis=0)
    projection_y = np.sum(cropped, axis=1)

    popt_x, _ = curve_fit(gaussian, x_vals_centered, projection_x)
    popt_y, _ = curve_fit(gaussian, y_vals_centered, projection_y)

    file_index = int(file[-5:-3])
    file_indices.append(file_index)
    x0_values.append(popt_x[1] + x_length / 2 + roi["xMain"][0])
    sigma_x_values.append(abs(popt_x[2]))
    y0_values.append(popt_y[1] + y_length / 2 + roi["yMain"][0])
    sigma_y_values.append(abs(popt_y[2]))


# Example
plt.plot(
    x_vals_centered + x_length / 2 + roi["xMain"][0], projection_x, label="X Projection"
)
plt.plot(
    x_vals_centered + x_length / 2 + roi["xMain"][0],
    gaussian(x_vals_centered, *popt_x),
    label="Gaussian Fit X",
    linestyle="--",
    linewidth=3,
)
plt.title(f"Gaussian Fit along x axis of image # {file[-5:-3]}")
# plt.savefig("gaussian_fit_x.png", dpi=600)
plt.show()

# ------- Plotting the time evolution of the fitted parameters -------

plt.rcParams.update({"font.size": 12})
plt.subplot(2, 2, 1)
plt.scatter(time_stamps, y0_values, label="y0")
plt.xlabel("Time (s)")
plt.ylabel(r"$\mu_y$")
plt.subplot(2, 2, 2)
plt.scatter(time_stamps, x0_values, label="x0")
plt.xlabel("Time (s)")
plt.ylabel(r"$\mu_x$")
plt.subplot(2, 2, 3)
plt.scatter(time_stamps, sigma_y_values, label="sigma_y")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_y$")
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.scatter(time_stamps, sigma_x_values, label="sigma_x")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_x$")
plt.tight_layout()
plt.subplots_adjust(top=0.92)
# plt.savefig("gaussian_fit_parameters.png", dpi=600)
plt.show()

# ------- Compute the acceleration of the atomic cloud -------

start = np.where(np.array(file_indices) == 10)[0][0]
new_indices = file_indices[start:]
t = dt * np.arange(len(new_indices)) + start_tof


# Fit quadratic function to x0 values
def quadratic(t, a, b, c):
    return a * t**2 + b * t + c


popt_x_accel, cov_x = curve_fit(quadratic, t, x0_values[start:])
popt_y_accel, cov_y = curve_fit(quadratic, t, y0_values[start:])

# Plotting the quadratic fits
plt.figure(figsize=(10, 5))
plt.rcParams.update({"font.size": 14})
plt.subplot(1, 2, 1)
plt.plot(t, x0_values[start:], "o", label=r"$\mu_x$ measured")
plt.plot(t, quadratic(t, *popt_x_accel), label="Quadratic Fit", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel(r"$\mu_x$")
plt.title(r"Quadratic Fit for $\mu_x(t)$")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(t, y0_values[start:], "o", label=r"$\mu_y$ measured")
plt.plot(t, quadratic(t, *popt_y_accel), label="Quadratic Fit", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel(r"$\mu_y$")
plt.title(r"Quadratic Fit for $\mu_y(t)$")
plt.legend()
# plt.savefig("quadratic_fit_parameters.png", dpi=600)
plt.tight_layout()

# ------- Compute pixel size from acceleration -------

# Estimating Pixel size
g = 9.81  # m/s^2
a_x = popt_x_accel[0]  # m/s^2
a_y = popt_y_accel[0]  # m/s^2

# Uncertainties
a_x_err = np.sqrt(cov_x[0, 0])
a_y_err = np.sqrt(cov_y[0, 0])

pixel_size_est = g / (2 * np.sqrt(a_x**2 + a_y**2)) * 1e6  # µm

partial_ax = -g * a_x / (2 * (a_x**2 + a_y**2) ** (3 / 2)) * 1e6  # µm
partial_ay = -g * a_y / (2 * (a_x**2 + a_y**2) ** (3 / 2)) * 1e6  # µ

pixel_size_error = np.sqrt((partial_ax**2 * a_x_err) + (partial_ay**2 * a_y_err))

print("\nEstimated pixel size:", pixel_size_est.round(6), "µm")
print("Pixel size error:", round(pixel_size_error, 6), "µm")

# ------ Compute Temperature from the evolution of sigma value ------

pixel_size = 3.45 * 1e-6  # in meters, actual pixel size
# pixel_size = pixel_size_est * 1e-6  # convert µm to meters

sigma_x_values_meters = np.array(sigma_x_values[start:]) * pixel_size
sigma_y_values_meters = np.array(sigma_y_values[start:]) * pixel_size

u = 1.66053906660e-27  # kg
m_85rb = 84.91178974 * u  # kg
k_B = 1.380649e-23  # J/K


def sigma_evolution(t, sigma_0, T):
    return np.sqrt(sigma_0**2 + (k_B * T / m_85rb) * t**2)


# Fit the sigma evolution
popt_sigma_x, cov_sigma_x = curve_fit(
    sigma_evolution, t, sigma_x_values_meters, p0=[0.1, 0], bounds=(0, [np.inf, np.inf])
)
popt_sigma_y, cov_sigma_y = curve_fit(
    sigma_evolution, t, sigma_y_values_meters, p0=[0.1, 0], bounds=(0, [np.inf, np.inf])
)

# Plotting the sigma evolution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, sigma_x_values_meters, "o", label=r"$\sigma_x$ measured")
plt.plot(t, sigma_evolution(t, *popt_sigma_x), label="Fit", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_x$ (m)")
plt.title(r"Sigma Evolution for $\sigma_x(t)$")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(t, sigma_y_values_meters, "o", label=r"$\sigma_y$ measured")
plt.plot(t, sigma_evolution(t, *popt_sigma_y), label="Fit", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_y$ (m)")
plt.title(r"Sigma Evolution for $\sigma_y(t)$")
plt.legend()
plt.tight_layout()
# plt.savefig("sigma_evolution.png", dpi=600)

T_opt = popt_sigma_x[1]
T_err = np.sqrt(cov_sigma_x[1][1])
print(f"\nEstimated temperature from x: {T_opt * 1e3:.6f} mK")
print(f"Uncertainty: {T_err * 1e3:.6f} mK")

T_opt = popt_sigma_y[1]
T_err = np.sqrt(cov_sigma_y[1][1])
print(f"\nEstimated temperature from y: {T_opt * 1e3:.6f} mK")
print(f"Uncertainty: {T_err * 1e3:.6f} mK")
