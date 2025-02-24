# analysis.r: Full Bayesian Analysis Pipeline

# Load Required Libraries
library(INLA)
library(tidyverse)
library(pracma) # For periodicity analysis

# Step 1: Define the Input Data
X <- seq(0, 10, length.out = 100)  # Example time points
data_observed <- sin(2 * pi * X / 3) + rnorm(100, sd = 0.2)  # Example observations

# Step 2: Create a Mesh for the SPDE Framework
mesh <- inla.mesh.1d(seq(min(X), max(X), length.out = 50))
spde <- inla.spde2.matern(mesh = mesh, alpha = 1)  # Matérn kernel

# Step 3: Fit the Models
# Fit Non-Oscillatory (OU) Model
stack_matern <- inla.stack(
  data = list(y = data_observed),
  A = list(inla.spde.make.A(mesh = mesh, loc = X), matrix(1, nrow = length(X), ncol = 1)),
  effects = list(list(spatial = 1:spde$n.spde), list(intercept = 1))
)

result_ou <- inla(
  formula = y ~ 1 + f(spatial, model = spde),
  data = inla.stack.data(stack_matern),
  control.predictor = list(A = inla.stack.A(stack_matern), compute = TRUE),
  control.compute = list(dic = TRUE, cpo = TRUE)
)

# Extract the projection matrix for observed locations
A_obs <- inla.spde.make.A(mesh = mesh, loc = X)

# Multiply the projection matrix with the fitted values to match observed locations
fitted_values_at_obs <- as.vector(A_obs %*% result_ou$summary.random$spatial$mean)
fitted_mean <- fitted_values_at_obs
fitted_lower <- as.vector(A_obs %*% result_ou$summary.random$spatial$`0.025quant`)
fitted_upper <- as.vector(A_obs %*% result_ou$summary.random$spatial$`0.975quant`)

# Combine data into a data frame
plot_data <- data.frame(
  X = X,
  Observed = data_observed,
  Fitted = fitted_mean,
  Lower = fitted_lower,
  Upper = fitted_upper
)

# Create the plot
ggplot(plot_data, aes(x = X)) +
  geom_point(aes(y = Observed), color = "blue", size = 2) +  # Observed data
  geom_line(aes(y = Fitted), color = "red", size = 1) +      # Fitted mean
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "red", alpha = 0.2) +  # CI
  labs(title = "OU Model Fit", x = "X", y = "Y") +
  theme_minimal() +
  theme(legend.position = "top")


# Define trainable period for the cosine basis
period_initial <- 3  # Initial guess for the period
cosine_basis <- matrix(cos(2 * pi * X / period_initial), ncol = 1)  # Column matrix

# Define the projection matrices and effects
stack_cosine <- inla.stack(
  data = list(y = data_observed),
  A = list(
    inla.spde.make.A(mesh = mesh, loc = X),  # For spatial effect
    matrix(1, nrow = length(X), ncol = 1),   # For intercept
    cosine_basis                              # For cosine effect
  ),
  effects = list(
    list(spatial = 1:spde$n.spde),           # Spatial latent effect
    list(intercept = 1),                    # Intercept effect
    list(cosine = 1)                        # Single latent effect for cosine basis
  )
)


result_ouosc <- inla(
  formula = y ~ 1 + f(spatial, model = spde) + cosine,
  data = inla.stack.data(stack_cosine),
  control.predictor = list(A = inla.stack.A(stack_cosine), compute = TRUE),
  control.compute = list(dic = TRUE, cpo = TRUE)
)

# Extract fitted values (posterior mean and credible intervals)
predictions <- result_ouosc$summary.fitted.values
fitted_mean <- predictions$mean
fitted_lower <- predictions$`0.025quant`
fitted_upper <- predictions$`0.975quant`

# Combine into a data frame
plot_data <- data.frame(
  X = X,
  Observed = data_observed,
  Fitted = fitted_mean[1:length(X)],  # Match observed data length
  Lower = fitted_lower[1:length(X)],
  Upper = fitted_upper[1:length(X)]
)


# Plot
ggplot(plot_data, aes(x = X)) +
  geom_point(aes(y = Observed), color = "blue", size = 2, alpha = 0.6) +  # Observed data
  geom_line(aes(y = Fitted), color = "red", size = 1) +                  # Fitted mean
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "red", alpha = 0.2) +  # Credible intervals
  labs(
    title = "OU+Oscillator Model Fit",
    x = "X (Time)",
    y = "Y (Observed and Fitted Values)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )


# Step 4: Generate Synthetic Data
posterior_samples <- inla.posterior.sample(1000, result_ou)
synths <- lapply(1:10, function(k) {
  posterior_pred <- posterior_samples[[k]]$latent
  posterior_pred[grep("Predictor", rownames(posterior_pred))]
})

# Step 5: Posterior Predictive Checks
ppl_observed <- mean(sapply(1:1000, function(i) {
  log_likelihood <- inla.compute.loglik(result_ou, data_observed)
  sum(log_likelihood)
}))

ppl_synth <- mean(sapply(synths, function(synth_data) {
  log_likelihood <- inla.compute.loglik(result_ou, synth_data)
  sum(log_likelihood)
}))

# Step 6: Compute Bayes Factors
log_marg_ou <- result_ou$mlik[1]
log_marg_ouosc <- result_ouosc$mlik[1]
bayes_factor <- exp(log_marg_ouosc - log_marg_ou)

# Step 7: Refine Bayes Factor Cutoffs
synth_bayes_factors <- sapply(synths, function(synth_data) {
  log_marg_synth_ou <- inla.compute.mlik(result_ou, synth_data)
  log_marg_synth_ouosc <- inla.compute.mlik(result_ouosc, synth_data)
  exp(log_marg_synth_ouosc - log_marg_synth_ou)
})

threshold <- quantile(synth_bayes_factors, 0.95)

# Step 8: Classify Cells
posterior_prob_oscillatory <- 1 / (1 + exp(log_marg_ou - log_marg_ouosc))
is_oscillatory <- posterior_prob_oscillatory > 0.95

# Step 9: Periodicity Detection
posterior_periods <- inla.posterior.sample(1000, result_ouosc)$summary.hyperpar["cosine.period", ]
mean_period <- mean(posterior_periods)
ci_period <- quantile(posterior_periods, c(0.025, 0.975))

# Output Results
cat("Bayes Factor:", bayes_factor, "\n")
cat("Threshold:", threshold, "\n")
cat("Posterior Probability of Oscillation:", posterior_prob_oscillatory, "\n")
cat("Mean Period:", mean_period, "\n")
cat("95% Credible Interval for Period:", ci_period, "\n")
