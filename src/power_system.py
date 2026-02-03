import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import config
import os

class PowerAwareCloudSystem:
    def __init__(self):
        self.model = None
        self.data = None
        self.results = []

    def load_and_preprocess_data(self):
        """
        Step 1: Loads the Cleaned CSV data.
        """
        print("[-] Step 1: Loading Preprocessed CSV Data...")

        from data_preprocessing import preprocess_csv_dataset

        cleaned_path = os.path.join(config.RESULTS_DIR, 'cleaned_cloud_data.csv')

        if not os.path.exists(cleaned_path):
            print("[-] Cleaned data not found. Running preprocessing script...")
            cleaned_path = preprocess_csv_dataset()

        try:
            self.data = pd.read_csv(cleaned_path, index_col=0)
            print(f"[+] Cleaned Data Loaded. Shape: {self.data.shape}")
            print(f"[+] Columns Available: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"[!] Error loading cleaned CSV: {e}")
            return False

    def generate_power_labels(self):
        """
        Generates synthetic power labels (PowerGen Logic).
        """
        print("[-] Generating Synthetic Power Consumption Labels...")

        # Sum utilization across all roles to get total system load
        total_load = self.data.sum(axis=1)

        # Generate Power Target
        power_consumption = config.BASE_POWER_WATTS + (total_load * config.CPU_POWER_COEFF)

        # Add slight random noise to make it realistic for AI to learn
        noise = np.random.normal(0, 2, size=len(power_consumption))
        self.data['Power_Consumption'] = power_consumption + noise

        print("[+] Power labels generated and added to dataset.")

    def train_ai_model(self):
        """
        Step 2: Build Power Consumption Prediction Model with Hyperparameter Tuning
        Uses Gradient Boosting Regressor.
        """
        print("[-] Step 2: Training AI Power Prediction Model...")

        X = self.data.drop('Power_Consumption', axis=1)
        y = self.data['Power_Consumption']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Hyperparameter Tuning using GridSearchCV
        from sklearn.model_selection import GridSearchCV

        model = GradientBoostingRegressor()

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        print(f"[+] Best Hyperparameters: {grid_search.best_params_}")

        # Use the best model found
        self.model = grid_search.best_estimator_

        # Evaluate the best model on test data
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"[+] Model Training Complete.")
        print(f"    - Mean Squared Error: {mse:.4f}")
        print(f"    - R^2 Score: {r2:.4f}")

    def run_simulation(self):
        """
        Step 3 & 4: Dynamic Scaling & Real-Time Monitoring
        """
        print("[-] Step 3 & 4: Running Dynamic Resource Scaling Simulation...")

        active_vms = 1  # Start with 1 VM
        logs = []

        simulation_data = self.data.head(100)

        for index, row in simulation_data.iterrows():
            current_features = pd.DataFrame([row.drop('Power_Consumption')],
                                            columns=row.drop('Power_Consumption').index)
            predicted_power = self.model.predict(current_features)[0]

            # Perform scaling decision based on predicted power
            self.scale_resources(predicted_power)

            # Track simulation results
            logs.append({
                'Time': index,
                'Predicted_Power': predicted_power,
                'Active_VMs': active_vms,
                'Action': 'SCALE'  # Placeholder for actual scaling action
            })

        self.sim_results = pd.DataFrame(logs)
        print("[+] Simulation Complete.")
        return self.sim_results

    def scale_resources(self, predicted_power):
        """
        Simulate resource scaling based on predicted power consumption (Locally).
        Instead of calling cloud APIs, we simulate scaling actions locally.
        """
        action = 'HOLD'  # Default action

        if predicted_power > config.HIGH_THRESHOLD:
            action = 'SCALE_UP'
            print("[+] Scaling up resources (Locally)...")
            self.update_vm_count(2)  # Simulate scaling up to 2 VMs
        elif predicted_power < config.IDLE_THRESHOLD:
            action = 'SCALE_DOWN'
            print("[+] Scaling down resources (Locally)...")
            self.update_vm_count(1)  # Simulate scaling down to 1 VM
        else:
            print("[+] Holding resources (Locally)...")

        return action

    def update_vm_count(self, desired_capacity):
        """
        Simulate the process of changing the number of active VMs (Locally).
        """
        print(f"[+] Active VM count set to: {desired_capacity}")

    def visualize_results(self):
        """
        Generates plots for the report.
        """
        print("[-] Generating Visualization...")
        plt.figure(figsize=(12, 8))

        # Plot predicted vs actual power consumption
        plt.subplot(2, 1, 1)
        plt.plot(self.sim_results['Time'], self.sim_results['Predicted_Power'], label='Predicted Power')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption')
        plt.title('Predicted Power Consumption')
        plt.legend()

        # Plot active VMs vs time
        plt.subplot(2, 1, 2)
        plt.plot(self.sim_results['Time'], self.sim_results['Active_VMs'], label='Active VMs', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Active VMs')
        plt.title('Active VMs Over Time')
        plt.legend()

        # Save the plot to the results folder
        output_path = os.path.join(config.RESULTS_DIR, 'power_management_results.png')
        plt.tight_layout()
        plt.savefig(output_path)

        print(f"[+] Graph saved to: {output_path}")

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Check if config is set
    if "PUT YOUR" in config.DATASET_PATH:
        print("\n[!] ERROR: Please edit config.py and put the correct path to your text file.\n")
    else:
        # Initialize System
        system = PowerAwareCloudSystem()

        # 1. Load Data
        if system.load_and_preprocess_data():
            # 2. Generate Power Labels (Simulating PowerGen)
            system.generate_power_labels()

            # 3. Train AI Model
            system.train_ai_model()

            # 4. Run Scaling Simulation
            results = system.run_simulation()

            # 5. Visualize & Save to 'results/' folder
            system.visualize_results()

            # 6. Show Sample Output
            print("\n--- Sample Simulation Output ---")
            print(results.head(10))