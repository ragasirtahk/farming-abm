from mesa import Agent
import numpy as np

class Farmer(Agent):
    def __init__(self, unique_id, model, land_size, initial_wealth, dissemination_mode):
        super().__init__(unique_id, model)
        self.land_size = land_size
        self.wealth = initial_wealth
        self.type = "Small" if self.land_size <= 2.5 else "Large"  # Farmer type
        self.trust_level = np.random.uniform(0.5, 1.0)
        self.crop_type = "none"
        self.previous_wealth = initial_wealth
        self.total_yield = 0  # Add this line to track total yield
        self.dissemination_mode = dissemination_mode

    def step(self):
        forecast = self.model.forecaster.provide_forecast()
        decision = self.decide_crop(forecast)
        yield_per_hectare = self.model.forecaster.get_yield(decision, forecast)
        earnings = yield_per_hectare * self.land_size
        self.wealth += earnings + self.model.subsidy_level
        self.total_yield += yield_per_hectare * self.land_size  # Update total yield
        self.update_trust(forecast)

    def decide_crop(self, forecast):
        if self.type == "Small":
            risk_tolerance = 0.6  # More cautious
        else:
            risk_tolerance = 0.8  # More risk-taking

        if forecast["rainfall"] > 50 * risk_tolerance:
            self.crop_type = "rice"
        else:
            self.crop_type = "wheat"
        return self.crop_type

    def update_trust(self, forecast):
        # Trust increases if wealth improves, decreases otherwise
        wealth_change_factor = (self.wealth - self.previous_wealth) / max(self.previous_wealth, 1)
        self.previous_wealth = self.wealth

        # Community trust factor
        community_trust = np.mean([
            neighbor.trust_level for neighbor in self.model.schedule.agents if isinstance(neighbor, Farmer)
        ])

        # Combine factors to adjust trust
        trust_change = 0.1 * wealth_change_factor + 0.1 * (community_trust - self.trust_level)
        if forecast["accuracy"] > 0.7:
            trust_change += 0.05  # Boost for accurate forecasts
        else:
            trust_change -= 0.1  # Penalty for low accuracy

        # Adjust trust change based on dissemination mode with updated probabilities
        if self.dissemination_mode == "Radio":
            trust_change *= 0.9  # Less trust due to impersonal nature
        elif self.dissemination_mode == "Mobile App":
            trust_change *= 1.0  # Baseline trust change
        elif self.dissemination_mode == "Extension Officer":
            trust_change *= 1.2  # Higher trust due to personal interaction
        elif self.dissemination_mode == "Community Leaders":
            trust_change *= 1.1  # Higher trust due to community influence

        self.trust_level = max(0.0, min(1.0, self.trust_level + trust_change))
        return self.trust_level

class Forecaster(Agent):
    def __init__(self, unique_id, model, accuracy, dissemination_mode="Radio"):
        super().__init__(unique_id, model)
        self.accuracy = accuracy
        self.dissemination_mode = dissemination_mode
        self.rainfall_variability = 10

    def provide_forecast(self):
        # Introduce a crisis: 5% chance of forecast failure
        if np.random.random() < 0.05:
            return {"rainfall": 0, "accuracy": 0}  # Failed forecast
        return {"rainfall": np.random.uniform(30, 70), "accuracy": self.adjusted_accuracy()}

    def adjusted_accuracy(self):
        if self.dissemination_mode == "Radio":
            return self.accuracy * 0.9  # Slightly lower accuracy due to general broadcast
        elif self.dissemination_mode == "Mobile App":
            return self.accuracy * 1.0  # Baseline accuracy
        elif self.dissemination_mode == "Extension Officer":
            return self.accuracy * 1.1  # Higher accuracy due to personalized advice
        elif self.dissemination_mode == "Community Leaders":
            return self.accuracy * 1.05  # Moderately higher accuracy due to trusted source
        return self.accuracy

    def get_yield(self, crop_type, forecast):
        if crop_type == "rice" and forecast["rainfall"] > 50:
            return np.random.uniform(2.0, 3.0) * forecast["accuracy"]
        elif crop_type == "wheat" and forecast["rainfall"] <= 50:
            return np.random.uniform(1.0, 2.0) * forecast["accuracy"]
        else:
            return np.random.uniform(0.5, 1.0) * forecast["accuracy"]