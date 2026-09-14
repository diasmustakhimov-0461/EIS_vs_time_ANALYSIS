
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from pathlib import Path


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "EIS_data.csv"
TARGET_FREQUENCY = 0.01

OUTPUT_FOLDER = Path("EIS_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 2. EQUIVALENT CIRCUITS
# ============================================================

def parallel_R_CPE(frequency, R, Q, n):
    """
    Parallel combination of a resistor and a CPE.

    Z = 1 / [1/R + Q*(j*omega)^n]
    """
    omega = 2 * np.pi * frequency
    Y = (1 / R) + Q * (1j * omega) ** n
    return 1 / Y


def circuit_one_time_constant(frequency, p):
    """
    One-time-constant circuit:

        Rs + (Rct || CPEdl)

    Parameters:
        p = [Rs, Rct, Qdl, ndl]
    """
    Rs, Rct, Qdl, ndl = p

    return Rs + parallel_R_CPE(
        frequency, Rct, Qdl, ndl
    )


def circuit_two_time_constants(frequency, p):
    """
    Two-time-constant circuit:

        Rs + (Rcoat || CPEcoat)
           + (Rct || CPEdl)

    Parameters:
        p = [Rs, Rcoat, Qcoat, ncoat, Rct, Qdl, ndl]
    """
    Rs, Rcoat, Qcoat, ncoat, Rct, Qdl, ndl = p

    Zcoat = parallel_R_CPE(
        frequency, Rcoat, Qcoat, ncoat
    )

    Zct = parallel_R_CPE(
        frequency, Rct, Qdl, ndl
    )

    return Rs + Zcoat + Zct


# ============================================================
# 3. RESIDUAL FUNCTIONS FOR FITTING
# ============================================================

def residual_one_time_constant(p, frequency, Z_exp):
    Z_model = circuit_one_time_constant(frequency, p)

    scale = np.maximum(np.abs(Z_exp), 1e-12)

    real_error = (Z_model.real - Z_exp.real) / scale
    imag_error = (Z_model.imag - Z_exp.imag) / scale

    return np.concatenate([real_error, imag_error])


def residual_two_time_constants(p, frequency, Z_exp):
    Z_model = circuit_two_time_constants(frequency, p)

    scale = np.maximum(np.abs(Z_exp), 1e-12)

    real_error = (Z_model.real - Z_exp.real) / scale
    imag_error = (Z_model.imag - Z_exp.imag) / scale

    return np.concatenate([real_error, imag_error])


# ============================================================
# 4. FIT ONE-TIME-CONSTANT MODEL
# ============================================================

def fit_one_time_constant(frequency, Z_exp):

    Rs0 = max(np.min(Z_exp.real), 1e-6)
    Rct0 = max(np.max(Z_exp.real) - Rs0, 1.0)

    p0 = [
        Rs0,
        Rct0,
        1e-5,   # Qdl
        0.85    # ndl
    ]

    lower = [
        0.0,
        1e-6,
        1e-12,
        0.3
    ]

    upper = [
        np.inf,
        np.inf,
        1.0,
        1.0
    ]

    result = least_squares(
        residual_one_time_constant,
        p0,
        args=(frequency, Z_exp),
        bounds=(lower, upper),
        max_nfev=50000
    )

    Z_fit = circuit_one_time_constant(
        frequency, result.x
    )

    return result, Z_fit


# ============================================================
# 5. FIT TWO-TIME-CONSTANT MODEL
# ============================================================

def fit_two_time_constants(frequency, Z_exp):

    Rs0 = max(np.min(Z_exp.real), 1e-6)

    total_R = max(
        np.max(Z_exp.real) - Rs0,
        10.0
    )

    Rcoat0 = total_R * 0.3
    Rct0 = total_R * 0.7

    p0 = [
        Rs0,
        Rcoat0,
        1e-5,   # Qcoat
        0.85,   # ncoat
        Rct0,
        1e-5,   # Qdl
        0.85    # ndl
    ]

    lower = [
        0.0,
        1e-6,
        1e-12,
        0.3,
        1e-6,
        1e-12,
        0.3
    ]

    upper = [
        np.inf,
        np.inf,
        1.0,
        1.0,
        np.inf,
        1.0,
        1.0
    ]

    result = least_squares(
        residual_two_time_constants,
        p0,
        args=(frequency, Z_exp),
        bounds=(lower, upper),
        max_nfev=50000
    )

    Z_fit = circuit_two_time_constants(
        frequency, result.x
    )

    return result, Z_fit


# ============================================================
# 6. BIC CALCULATION
# ============================================================

def calculate_bic(residual, number_of_parameters):

    n = len(residual)
    rss = np.sum(residual ** 2)
    rss = max(rss, 1e-300)

    return (
        n * np.log(rss / n)
        + number_of_parameters * np.log(n)
    )


# ============================================================
# 7. READ DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = [
    "immersion_time",
    "frequency",
    "Zreal",
    "Zimag"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

df = df.dropna(subset=required_columns)

df = df.sort_values(
    ["immersion_time", "frequency"],
    ascending=[True, False]
)


# ============================================================
# 8. FIT EACH IMMERSION TIME
# ============================================================

results_list = []
fitted_spectra_list = []

immersion_times = sorted(
    df["immersion_time"].unique()
)

print("\nStarting EIS analysis...\n")

for time in immersion_times:

    data = df[
        df["immersion_time"] == time
    ].copy()

    frequency = data["frequency"].to_numpy(
        dtype=float
    )

    Z_exp = (
        data["Zreal"].to_numpy(dtype=float)
        + 1j * data["Zimag"].to_numpy(dtype=float)
    )

    # Keep only positive frequencies
    valid = frequency > 0

    frequency = frequency[valid]
    Z_exp = Z_exp[valid]

    # Sort from high to low frequency
    order = np.argsort(frequency)[::-1]

    frequency = frequency[order]
    Z_exp = Z_exp[order]

    if len(frequency) < 5:
        print(
            f"Skipping time {time}: "
            "not enough data points."
        )
        continue

    # --------------------------------------------------------
    # Fit both models
    # --------------------------------------------------------

    fit1, Zfit1 = fit_one_time_constant(
        frequency, Z_exp
    )

    fit2, Zfit2 = fit_two_time_constants(
        frequency, Z_exp
    )

    residual1 = residual_one_time_constant(
        fit1.x, frequency, Z_exp
    )

    residual2 = residual_two_time_constants(
        fit2.x, frequency, Z_exp
    )

    bic1 = calculate_bic(
        residual1, len(fit1.x)
    )

    bic2 = calculate_bic(
        residual2, len(fit2.x)
    )

    # --------------------------------------------------------
    # Select the model with lower BIC
    # --------------------------------------------------------

    if bic2 < bic1:

        selected_model = "two_time_constants"

        p = fit2.x
        Z_fit = Zfit2

        Rs = p[0]
        Rcoat = p[1]
        Qcoat = p[2]
        ncoat = p[3]
        Rct = p[4]
        Qdl = p[5]
        ndl = p[6]

    else:

        selected_model = "one_time_constant"

        p = fit1.x
        Z_fit = Zfit1

        Rs = p[0]
        Rcoat = np.nan
        Qcoat = np.nan
        ncoat = np.nan
        Rct = p[1]
        Qdl = p[2]
        ndl = p[3]

    # --------------------------------------------------------
    # Calculate RMSE
    # --------------------------------------------------------

    residual_complex = Z_fit - Z_exp

    rmse = np.sqrt(
        np.mean(np.abs(residual_complex) ** 2)
    )

    # --------------------------------------------------------
    # Calculate |Z| at 0.01 Hz
    # --------------------------------------------------------

    log_f = np.log10(frequency)
    log_target = np.log10(TARGET_FREQUENCY)

    if (
        TARGET_FREQUENCY >= frequency.min()
        and TARGET_FREQUENCY <= frequency.max()
    ):

        # Interpolate experimental |Z|
        log_abs_Z = np.log10(np.abs(Z_exp))

        abs_Z_target = 10 ** np.interp(
            log_target,
            log_f[::-1],
            log_abs_Z[::-1]
        )

        source = "experimental interpolation"

    else:

        # Use fitted model if target is outside range
        if selected_model == "two_time_constants":
            Z_target = circuit_two_time_constants(
                np.array([TARGET_FREQUENCY]), p
            )[0]
        else:
            Z_target = circuit_one_time_constant(
                np.array([TARGET_FREQUENCY]), p
            )[0]

        abs_Z_target = np.abs(Z_target)
        source = "model extrapolation"

    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------

    results_list.append({
        "immersion_time": time,
        "selected_model": selected_model,
        "BIC_one_time_constant": bic1,
        "BIC_two_time_constants": bic2,
        "RMSE": rmse,
        "Rs": Rs,
        "Rcoat": Rcoat,
        "Rct": Rct,
        "Qcoat": Qcoat,
        "ncoat": ncoat,
        "Qdl": Qdl,
        "ndl": ndl,
        "abs_Z_at_0.01_Hz": abs_Z_target,
        "Z_target_source": source
    })

    # --------------------------------------------------------
    # Save measured and fitted spectra
    # --------------------------------------------------------

    for f, z_exp, z_fit in zip(
        frequency, Z_exp, Z_fit
    ):

        fitted_spectra_list.append({
            "immersion_time": time,
            "frequency": f,
            "Zreal_experimental": z_exp.real,
            "Zimag_experimental": z_exp.imag,
            "Zreal_fitted": z_fit.real,
            "Zimag_fitted": z_fit.imag
        })

    print(f"Immersion time: {time}")
    print(f"  Selected model: {selected_model}")
    print(f"  BIC (1 TC): {bic1:.3f}")
    print(f"  BIC (2 TC): {bic2:.3f}")
    print(f"  Rs = {Rs:.4g} ohm")
    print(f"  Rcoat = {Rcoat:.4g} ohm")
    print(f"  Rct = {Rct:.4g} ohm")
    print(
        f"  |Z| at 0.01 Hz = "
        f"{abs_Z_target:.4g} ohm"
    )
    print()


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results_list)

fitted_spectra_df = pd.DataFrame(
    fitted_spectra_list
)

results_df.to_csv(
    OUTPUT_FOLDER / "EIS_fitting_results.csv",
    index=False
)

fitted_spectra_df.to_csv(
    OUTPUT_FOLDER / "EIS_fitted_spectra.csv",
    index=False
)

print("Results saved in:", OUTPUT_FOLDER)


# ============================================================
# 10. NYQUIST PLOT
# ============================================================

if not fitted_spectra_df.empty:

    plt.figure(figsize=(8, 6))

    for time in immersion_times:

        data = fitted_spectra_df[
            fitted_spectra_df["immersion_time"] == time
        ]

        if data.empty:
            continue

        plt.plot(
            data["Zreal_experimental"],
            -data["Zimag_experimental"],
            "o",
            markersize=4,
            label=f"{time}"
        )

        plt.plot(
            data["Zreal_fitted"],
            -data["Zimag_fitted"],
            "-",
            linewidth=1.5
        )

    plt.xlabel(r"$Z'$ ($\Omega$)")
    plt.ylabel(r"$-Z''$ ($\Omega$)")
    plt.title("EIS spectra and equivalent-circuit fits")
    plt.legend(title="Immersion time")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "Nyquist_fits.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 11. |Z| AT 0.01 Hz VS IMMERSION TIME
# ============================================================

if not results_df.empty:

    plot_df = results_df.sort_values(
        "immersion_time"
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        plot_df["immersion_time"],
        plot_df["abs_Z_at_0.01_Hz"],
        "o-"
    )

    plt.xlabel("Immersion time")
    plt.ylabel(r"$|Z|$ at 0.01 Hz ($\Omega$)")
    plt.title(r"Evolution of $|Z|_{0.01\,Hz}$")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "Impedance_0.01Hz_vs_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 12. Rcoat VS IMMERSION TIME
# ============================================================

if not results_df.empty:

    plot_df = results_df.sort_values(
        "immersion_time"
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        plot_df["immersion_time"],
        plot_df["Rcoat"],
        "o-"
    )

    plt.xlabel("Immersion time")
    plt.ylabel(r"$R_{coat}$ ($\Omega$)")
    plt.title(r"Evolution of $R_{coat}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "Rcoat_vs_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 13. Rct VS IMMERSION TIME
# ============================================================

if not results_df.empty:

    plot_df = results_df.sort_values(
        "immersion_time"
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        plot_df["immersion_time"],
        plot_df["Rct"],
        "o-"
    )

    plt.xlabel("Immersion time")
    plt.ylabel(r"$R_{ct}$ ($\Omega$)")
    plt.title(r"Evolution of $R_{ct}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "Rct_vs_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


print("\nAnalysis completed successfully.")
