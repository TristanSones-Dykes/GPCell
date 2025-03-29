library(tidyverse)


# Define parameters
n_cells <- 1000 # total replicates used in the simulation
t_final_vals <- c(1500, 600) # 25/10 hours
noise_vals <- c(0.1, 0.5)

# Build simulation parameters:
# For each noise value, create a parameter vector with the first time value
sim_params <- lapply(noise_vals, function(noise) c(noise, t_final_vals[1], n_cells))
# Append an extra simulation parameter using the first noise value and the second time value
sim_params <- c(sim_params, list(c(noise_vals[1], t_final_vals[2], n_cells)))

# Define the base path for the simulation data
path <- "data/matlab/"

# Create the list of file paths using the simulation parameters
paths <- sapply(sim_params, function(param) {
  noise <- param[1]
  t_final <- param[2]
  n_cells <- param[3]
  sprintf("%sdataNORMED%s_%.1f_%s.csv", path, t_final, noise, n_cells)
})

# Time columns
time_cols <- list()
for (i in seq_along(sim_params)) {
  time_cols[[i]] <- seq(0, sim_params[[i]][2] / 60, 0.5)
}

# read in each file, add time column "Time", name rest of columns "Cell 1", "Cell 2", etc.
data_list <- lapply(seq_along(paths), function(i) {
  data <- read.csv(paths[i], header = FALSE)
  colnames(data) <- paste0("Cell ", seq_len(ncol(data)))

  # add time column, place it first
  data <- data %>%
    mutate(Time = time_cols[[i]]) %>%
    select(Time, everything())

  # save csv
  write.csv(data,
    file = sprintf(
      "data/matlab/noise_%.1f_time_%s_rep_%s.csv",
      sim_params[[i]][1], sim_params[[i]][2], sim_params[[i]][3]
    ),
    row.names = FALSE
  )

  # return the data frame
  data
})
