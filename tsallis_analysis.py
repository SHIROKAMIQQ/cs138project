import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit



class TsallisEarthquakeDistribution:
    
    def __init__(self, q=1.5, beta_s=0.1, beta_t=0.1):
        self.q = q
        self.beta_s = beta_s
        self.beta_t = beta_t
        
    def q_exponential(self, x, q=None):
        #e_q(x) = [1 + (1-q)x]_{+}^{1/(1-q)}
        #where [y]_+ = max(y, 0)
        if q is None:
            q = self.q
            
        if q == 1:
            return np.exp(x)
        
        exponent = 1.0 / (1.0 - q)
        inner = 1.0 + (1.0 - q) * x
        inner = np.maximum(inner, 0)
        return np.power(inner, exponent)
    
    def survival_distance(self, r):
        #P_>(r) = e_q(-beta_s * r)
        return self.q_exponential(-self.beta_s * r)
    
    def survival_time(self, t):
        #P_>(t) = e_q(-beta_t * t)
        return self.q_exponential(-self.beta_t * t)
    
    def pdf_distance(self, r):
        #p(r) = -d/dr [P_>(r)]
        if self.q == 1:
            # Exponential distribution
            return self.beta_s * np.exp(-self.beta_s * r)
        else:
            # Tsallis q-exponential distribution
            exponent = (2.0 - self.q) / (1.0 - self.q)
            inner = 1.0 + (self.q - 1.0) * self.beta_s * r
            inner = np.maximum(inner, 0)
            return self.beta_s * np.power(inner, -exponent)
    
    def pdf_time(self, t):
        #p(t) = -d/dt [P_≥(t)]
        if self.q == 1:
            # Exponential distribution
            return self.beta_t * np.exp(-self.beta_t * t)
        else:
            # Tsallis q-exponential distribution
            exponent = (2.0 - self.q) / (1.0 - self.q)
            inner = 1.0 + (self.q - 1.0) * self.beta_t * t
            inner = np.maximum(inner, 0)
            return self.beta_t * np.power(inner, -exponent)

class TsallisFitter:
   
    def __init__(self):
        self.distance_model = None
        self.time_model = None
        
    def calculate_distances(self, df, epicenter_lat, epicenter_lon):
        #calculate distances from epicenter for all earthquakes
        
        # Convert to radians
        lat1 = np.radians(epicenter_lat)
        lon1 = np.radians(epicenter_lon)
        lat2 = np.radians(df['latitude'].values)
        lon2 = np.radians(df['longitude'].values)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r_earth = 6371.0
        distances = c * r_earth
        
        return distances
    
    def calculate_time_intervals(self, df, time_col='updated'):
        # calculate time intervals between consecutive earthquakes (calm times)
                
        # Calculate time differences
        time_diffs = df[time_col].diff().dropna()
        
        # Convert to days
        calm_times = time_diffs.dt.total_seconds() / (24 * 3600)
        
        return calm_times.values
    
    def fit_distance_distribution(self, distances, q_guess=1.5, beta_guess=0.1):
        # fit distribution to distance data

        # Filter out zero distances if any
        distances = np.array(distances)
        distances = distances[distances > 0]
        
        # Define objective function for survival distribution
        def survival_func(r, q, beta):
            temp_model = TsallisEarthquakeDistribution(q=q, beta_s=beta)
            return temp_model.survival_distance(r)
        
        # Fit using curve_fit on survival function
        # Create empirical survival function
        n = len(distances)
        sorted_dists = np.sort(distances)
        empirical_survival = 1 - np.arange(1, n + 1) / n
        
        # Initial parameter guesses
        p0 = [q_guess, beta_guess]
        
        # Bounds: q > 1 (for power-law tail), beta > 0
        bounds = ([1.0001, 1e-10], [5, 10])
        
        try:
            popt, pcov = curve_fit(
                survival_func, 
                sorted_dists, 
                empirical_survival,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            q_opt, beta_opt = popt
            self.distance_model = TsallisEarthquakeDistribution(
                q=q_opt, 
                beta_s=beta_opt
            )
            
            print(f"Fitted distance distribution:")
            print(f"  q = {q_opt:.4f}")
            print(f"  beta_s = {beta_opt:.4f}")
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            print("Using initial guesses")
            self.distance_model = TsallisEarthquakeDistribution(
                q=q_guess, 
                beta_s=beta_guess
            )
        
        return self.distance_model
    
    def fit_time_distribution(self, calm_times, q_guess=1.5, beta_guess=0.1):
        # fit Tsallis distribution to calm time data
        
        # Filter out zero or negative times
        calm_times = np.array(calm_times)
        calm_times = calm_times[calm_times > 0]
        
        # Define objective function
        def survival_func(t, q, beta):
            temp_model = TsallisEarthquakeDistribution(q=q, beta_t=beta)
            return temp_model.survival_time(t)
        
        # Create empirical survival function
        n = len(calm_times)
        sorted_times = np.sort(calm_times)
        empirical_survival = 1 - np.arange(1, n + 1) / n
        
        # Initial parameter guesses
        p0 = [q_guess, beta_guess]
        
        # Bounds
        bounds = ([1.0001, 1e-10], [5, 10])
        
        try:
            popt, pcov = curve_fit(
                survival_func,
                sorted_times,
                empirical_survival,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            q_opt, beta_opt = popt
            self.time_model = TsallisEarthquakeDistribution(
                q=q_opt,
                beta_t=beta_opt
            )
            
            print(f"Fitted time distribution:")
            print(f"  q = {q_opt:.4f}")
            print(f"  beta_t = {beta_opt:.4f}")
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            print("Using initial guesses")
            self.time_model = TsallisEarthquakeDistribution(
                q=q_guess,
                beta_t=beta_guess
            )
        
        return self.time_model
    
    def goodness_of_fit(self, data, model_type='distance'):
        """
        Calculate goodness-of-fit statistics.
        """
        if model_type == 'distance' and self.distance_model:
            model = self.distance_model
            cdf_func = model.cdf_distance
        elif model_type == 'time' and self.time_model:
            model = self.time_model
            cdf_func = model.cdf_time
        else:
            raise ValueError("Model not fitted")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(
            data, 
            cdf_func
        )
        
        # Calculate AIC (Akaike Information Criterion)
        n = len(data)
        k = 2  # q and beta parameters
        
        # Calculate log-likelihood
        if model_type == 'distance':
            pdf_vals = model.pdf_distance(data)
        else:
            pdf_vals = model.pdf_time(data)
        
        # Avoid log(0)
        pdf_vals = np.maximum(pdf_vals, 1e-10)
        log_likelihood = np.sum(np.log(pdf_vals))
        
        aic = 2 * k - 2 * log_likelihood
        
        return {
            'KS_statistic': ks_stat,
            'KS_pvalue': ks_pvalue,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'n_parameters': k,
            'n_samples': n
        }


def plot_tsallis_fits(distances, times, distance_model, time_model):

    # plot Tsallis distribution fits.

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # crucial to avoid divide by zero
    distances = np.array(distances)
    distances = distances[distances > 0]
    times = np.array(times)
    times = times[times > 0]

    # Distance survival plot
    ax = axes[0, 0]
    sorted_dists = np.sort(distances)
    empirical_surv = 1 - np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
    
    ax.loglog(sorted_dists, empirical_surv, 'b.', alpha=0.6, 
              label='Empirical')
    
    dist_range = np.logspace(np.log10(sorted_dists[0]), 
                            np.log10(sorted_dists[-1]), 100)
    model_surv = distance_model.survival_distance(dist_range)
    
    ax.loglog(dist_range, model_surv, 'r-', linewidth=2, 
              label=f'Tsallis (q={distance_model.q:.2f})')
    
    ax.set_xlabel('Distance from epicenter (km)')
    ax.set_ylabel('P(≥ r)')
    ax.set_title('Distance Survival Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance PDF plot
    ax = axes[0, 1]
    hist, bins = np.histogram(distances, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.semilogy(bin_centers, hist, 'b.', alpha=0.6, label='Histogram')
    
    dist_range = np.linspace(0, distances.max(), 200)
    model_pdf = distance_model.pdf_distance(dist_range)
    
    ax.semilogy(dist_range, model_pdf, 'r-', linewidth=2, 
                label='Tsallis PDF')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distance PDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time survival plot
    ax = axes[1, 0]
    sorted_times = np.sort(times)
    empirical_surv = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    
    ax.loglog(sorted_times, empirical_surv, 'g.', alpha=0.6, 
              label='Empirical')
    
    time_range = np.logspace(np.log10(sorted_times[0]), 
                            np.log10(sorted_times[-1]), 100)
    model_surv = time_model.survival_time(time_range)
    
    ax.loglog(time_range, model_surv, 'r-', linewidth=2, 
              label=f'Tsallis (q={time_model.q:.2f})')
    
    ax.set_xlabel('Time interval (days)')
    ax.set_ylabel('P(≥ t)')
    ax.set_title('Time Interval Survival Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time PDF plot
    ax = axes[1, 1]
    hist, bins = np.histogram(times, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.semilogy(bin_centers, hist, 'g.', alpha=0.6, label='Histogram')
    
    time_range = np.linspace(0, times.max(), 200)
    model_pdf = time_model.pdf_time(time_range)
    
    ax.semilogy(time_range, model_pdf, 'r-', linewidth=2, 
                label='Tsallis PDF')
    
    ax.set_xlabel('Time interval (days)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Time Interval PDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig

def setup1():
    #using sample size as is to produce q and beta_s values that fit the distribution.
    print("Using fitting")
    TED = TsallisEarthquakeDistribution()
    TF = TsallisFitter()
    distances = TF.calculate_distances(distrb_points,arb_lat,arb_lon)
    calms = TF.calculate_time_intervals(distrb_points)
    pprint(calms)
    pprint(distances)

    TF.fit_distance_distribution(distances)
    TF.fit_time_distribution(calms)

    plot_tsallis_fits(distances, calms, TF.distance_model, TF.time_model)


# extract usgs input and set sample size
INPUT_FILE = "202312Spacial.csv"
N =100

# Load .csv file and get relevant points, update updated for suitable datetime
df = pd.read_csv(INPUT_FILE)
points = df[['latitude', 'longitude', 'mag', 'updated']]
distrb_points = points.sample(n=N)
distrb_points['updated'] = pd.to_datetime(distrb_points['updated'])
distrb_points = distrb_points.sort_values('updated')

arb_lat = distrb_points["latitude"].iloc[0]
arb_lon = distrb_points["longitude"].iloc[0]
print(f"designated epicenter: LAT {arb_lat}, LON {arb_lon}")

setup1()