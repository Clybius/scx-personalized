// Phase 2: Profile configurations
// gaming, productivity, server profiles with different loss weights

use std::str::FromStr;

pub struct LossWeights {
    pub latency: f64,
    pub throughput: f64,
    pub fairness: f64,
}

impl Clone for LossWeights {
    fn clone(&self) -> Self {
        Self {
            latency: self.latency,
            throughput: self.throughput,
            fairness: self.fairness,
        }
    }
}

pub struct Profile {
    pub name: String,
    pub weights: LossWeights,
    pub learning_rate: f64,
    pub update_interval_ms: u64,
}

impl Profile {
    pub fn gaming() -> Self {
        Self {
            name: "gaming".to_string(),
            weights: LossWeights {
                latency: 0.8,
                throughput: 0.15,
                fairness: 0.05,
            },
            learning_rate: 0.01,
            update_interval_ms: 50,
        }
    }

    pub fn productivity() -> Self {
        Self {
            name: "productivity".to_string(),
            weights: LossWeights {
                latency: 0.4,
                throughput: 0.4,
                fairness: 0.2,
            },
            learning_rate: 0.005,
            update_interval_ms: 100,
        }
    }

    pub fn server() -> Self {
        Self {
            name: "server".to_string(),
            weights: LossWeights {
                latency: 0.2,
                throughput: 0.5,
                fairness: 0.3,
            },
            learning_rate: 0.002,
            update_interval_ms: 200,
        }
    }

    /// NEW: Phase 3 - Get loss weights for optimizer
    pub fn loss_weights(&self) -> LossWeights {
        self.weights.clone()
    }
}

pub struct ProfileConfig {
    pub profile: Profile,
}

impl ProfileConfig {
    pub fn from_name(name: &str) -> Self {
        let profile = match name {
            "gaming" => Profile::gaming(),
            "server" => Profile::server(),
            _ => Profile::productivity(), // default
        };

        Self { profile }
    }
}

impl FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gaming" => Ok(Self::gaming()),
            "productivity" => Ok(Self::productivity()),
            "server" => Ok(Self::server()),
            _ => Err(format!("Unknown profile: {}", s)),
        }
    }
}
