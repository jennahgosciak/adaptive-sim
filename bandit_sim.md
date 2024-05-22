Bandit Simulation for ECON 7245
================
Jennah Gosciak

This program simulates different Multi-Armed Bandit (MAB) algorithms in
an experiment to detect age discrimination against political candidates.
One can view the performance of the algorithms under different
probabilities of discriminating, see .

``` r
library(foreach)
```

    ## Warning: package 'foreach' was built under R version 4.2.3

``` r
library(doParallel)
```

    ## Warning: package 'doParallel' was built under R version 4.2.3

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 4.2.3

    ## Loading required package: parallel

``` r
library(tidyverse)
```

    ## ── Attaching packages
    ## ───────────────────────────────────────
    ## tidyverse 1.3.2 ──

    ## ✔ ggplot2 3.4.4      ✔ purrr   1.0.2 
    ## ✔ tibble  3.1.8      ✔ dplyr   1.0.10
    ## ✔ tidyr   1.3.1      ✔ stringr 1.5.1 
    ## ✔ readr   2.1.3      ✔ forcats 1.0.0

    ## Warning: package 'ggplot2' was built under R version 4.2.3

    ## Warning: package 'tidyr' was built under R version 4.2.3

    ## Warning: package 'purrr' was built under R version 4.2.3

    ## Warning: package 'stringr' was built under R version 4.2.3

    ## Warning: package 'forcats' was built under R version 4.2.3

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ purrr::accumulate() masks foreach::accumulate()
    ## ✖ dplyr::filter()     masks stats::filter()
    ## ✖ dplyr::lag()        masks stats::lag()
    ## ✖ purrr::when()       masks foreach::when()

``` r
library(purrr)

# Identify num. cores in CPU
n_cores <- detectCores()
n_cores
```

    ## [1] 8

``` r
# Register cluster
cluster <- makeCluster(n_cores - 1)
registerDoParallel(cluster)

# probability of choosing the younger candidate in each of the 8 contexts
# context 4 is the most discriminatory
context_prob <- c(rep(0.5, 7), 0.7)
# number of survey respondents to vary
obs <- seq(100, 500, 100)
```

# Explore-then-commit Algorithm

``` r
## explore then commit algorithm
etc_algorithm <- function(prob, n, m=NULL,
                          reps=100, k=8) {
  
  # Keep track of the start time
  time_start <- Sys.time()

  # Create and register cluster
  cluster <- makeCluster(n_cores)
  registerDoParallel(cluster)

  # Only for measuring computation time
  time_start_processing <- Sys.time()
  
  total_regret <- foreach (r = 1:reps, .combine = "c")  %dopar% {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    
    # explore
    for (t in 1:(m*k)) {
      a_t <- t%%k + 1
      a[t] <- a_t
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    mu_mean <- 
      apply(mu[1:(m*k),], 2, 
            mean, na.rm = TRUE)
    # exploit
    for (t in ((m*k)+1):n) {
          a_t <- which.max(mu_mean)
          a[t] <- a_t
          regret[t] <- max(prob) - prob[a_t]
    }
    return(sum(regret))
  }
  
  # Only for measuring computation time
  time_finish_processing <- Sys.time()

  # Stop the cluster
  stopCluster(cl = cluster)

  # Keep track of the end time
  time_end <- Sys.time()
  
  print(time_end - time_start)
  
  return(mean(total_regret))
}
```

``` r
regret_val <- c()
m_vals <- c(1, 2, 5, 10, 20, 30, 40)
for (m in m_vals) {
  r <- etc_algorithm(context_prob, 1000, m = m)
  regret_val <- c(regret_val, r)
}
```

    ## Time difference of 1.799768 secs
    ## Time difference of 1.933743 secs
    ## Time difference of 1.474337 secs
    ## Time difference of 1.477391 secs
    ## Time difference of 2.260576 secs
    ## Time difference of 1.824782 secs
    ## Time difference of 1.595731 secs

``` r
ggplot(data = data.frame("m" = m_vals,
                         "expected regret" = regret_val),
       aes(m, `expected.regret`)) +
  geom_point() +
  theme_classic()
```

![](bandit_sim_v2_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# for fixed n, varying n
regret_val_etc <- c()
for (t in obs) {
  r <- etc_algorithm(context_prob, 
                     t, m = 2)
  regret_val_etc <- c(regret_val_etc, r)
}
```

    ## Time difference of 1.600549 secs
    ## Time difference of 1.844743 secs
    ## Time difference of 1.797663 secs
    ## Time difference of 1.560269 secs
    ## Time difference of 1.414009 secs

## Epsilon Greedy Algorithm

``` r
# epsilon-greedy algorithm
epsilon_greedy_algorithm <- function(prob, n, eps,
                          reps=100, k=8) {
  
  # Keep track of the start time
  time_start <- Sys.time()

  # Create and register cluster
  cluster <- makeCluster(n_cores)
  registerDoParallel(cluster)

  # Only for measuring computation time
  time_start_processing <- Sys.time()
  
  total_regret <- foreach (r = 1:reps, .combine = "c") %dopar% {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    
    # pull each arm once
    for (t in 1:k) {
      a_t <- t
      a[t] <- a_t
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    for (t in (k+1):n) {
      # if prob > eps, use the prior mean values
      if (runif(1) > eps) {
        mu_mean <- 
          colMeans(mu[1:(t-1),], na.rm = TRUE)
        a_t <- which.max(mu_mean)
      # otherwise, randomly explore
      } else {
        a_t <- sample(c(1:k), 1)
      }
      a[t] <- a_t
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    return(sum(regret))
  }
  
  # Only for measuring computation time
  time_finish_processing <- Sys.time()

  # Stop the cluster
  stopCluster(cl = cluster)

  # Keep track of the end time
  time_end <- Sys.time()
  
  print(time_end - time_start)
  
  return(mean(total_regret))
}
```

``` r
regret_val_eg <- c()
for (t in obs) {
  r <- epsilon_greedy_algorithm(context_prob, t, 
                                eps=0.1)
  regret_val_eg <- c(regret_val_eg, r)
}
```

    ## Time difference of 1.556255 secs
    ## Time difference of 2.057519 secs
    ## Time difference of 1.788565 secs
    ## Time difference of 1.989586 secs
    ## Time difference of 2.487376 secs

## UCB

``` r
## upper confidence bound
ucb_algorithm <- function(prob, n, delta,
                          reps=100, k=8) {

  # Keep track of the start time
  time_start <- Sys.time()

  # Create and register cluster
  cluster <- makeCluster(n_cores)
  registerDoParallel(cluster)

  # Only for measuring computation time
  time_start_processing <- Sys.time()
  
  total_regret <- foreach (r = 1:reps, .combine = "c") %dopar% {
    sink('log.txt', append=FALSE)
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    pulls <- vector("integer", k)
    ucb <- matrix(nrow=n, ncol=k)
    
    # first pull each arm once
    for (t in 1:k) {
      a_t <- t
      a[t] <- a_t
      pulls[a_t] <- pulls[a_t] + 1
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    for (t in (k+1):n) {
      mu_mean <- colMeans(mu[1:(t-1),], na.rm = TRUE)
      print(mu_mean)
      ucb <- mu_mean + delta*sqrt((1.5 * log(t)) / pulls)
      a_t <- which.max(ucb)
      print(a_t)
      
      a[t] <- a_t
      pulls[a_t] <- pulls[a_t] + 1
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    return(sum(regret))
  }
  
  # Only for measuring computation time
  time_finish_processing <- Sys.time()

  # Stop the cluster
  stopCluster(cl = cluster)

  # Keep track of the end time
  time_end <- Sys.time()
  
  print(time_end - time_start)
  
  return(mean(total_regret))
}
```

``` r
regret_val_ucb <- c()
for (t in obs) {
  r <- ucb_algorithm(context_prob, t, delta=0.8)
  regret_val_ucb <- c(regret_val_ucb, r)
}
```

    ## Time difference of 2.625868 secs
    ## Time difference of 4.14099 secs
    ## Time difference of 6.23347 secs
    ## Time difference of 6.498219 secs
    ## Time difference of 7.480465 secs

``` r
regret_val_ucb
```

    ## [1] 15.368 28.504 40.380 52.834 60.540

## MOSS

``` r
moss_algorithm <- function(prob, n, delta,
                          reps=100, k=8) {
  
  # Keep track of the start time
  time_start <- Sys.time()

  # Create and register cluster
  cluster <- makeCluster(n_cores)
  registerDoParallel(cluster)

  # Only for measuring computation time
  time_start_processing <- Sys.time()
  
  total_regret <- foreach (r = 1:reps, .combine = "c")  %dopar% {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    pulls <- vector("integer", k)
    
    # first choose each arm once
    for (i in 1:k) {
      a_t <- i
      pulls[a_t] <- pulls[a_t] + 1
      mu[i, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[i] <- max(prob) - prob[a_t]
    }
    
    # then choose arms in the following way
    for (t in (k+1):n) {
      mu_mean <- apply(mu[1:(t-1),], 2, 
                       mean, na.rm = TRUE)
      moss_bound <- mu_mean + sqrt((4 / pulls) * log(max(1, (n / k * pulls))))
      a_t <- which.max(moss_bound)
     
      a[t] <- a_t
      pulls[a_t] <- pulls[a_t] + 1
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    return(sum(regret))
  }
  
  # Only for measuring computation time
  time_finish_processing <- Sys.time()

  # Stop the cluster
  stopCluster(cl = cluster)

  # Keep track of the end time
  time_end <- Sys.time()
  
  print(time_end - time_start)
  
  return(mean(total_regret))
}
```

``` r
regret_val_moss <- c()
for (t in obs) {
  r <- moss_algorithm(context_prob, t, delta=0.1)
  regret_val_moss <- c(regret_val_moss, r)
}
```

    ## Time difference of 2.321989 secs
    ## Time difference of 3.236409 secs
    ## Time difference of 3.909908 secs
    ## Time difference of 5.20459 secs
    ## Time difference of 5.844072 secs

## Thompson Sampling

``` r
ts_algorithm <- function(prob, n,
                          reps=10, k=8) {
  # Keep track of the start time
  time_start <- Sys.time()

  # Create and register cluster
  cluster <- makeCluster(n_cores)
  registerDoParallel(cluster)

  # Only for measuring computation time
  time_start_processing <- Sys.time()
  
  #total_regret <- vector("double", reps)
  total_regret <- foreach (r = 1:reps, .combine = "c") %dopar% {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    alpha <- vector("integer", k)
    beta <- vector("integer", k)
    
    for (t in 1:n)  {
      # Monte Carlo simulation to estimate the probability of the best arm
      sim <- purrr::map2_dfc(alpha + 1, beta + 1, ~rbeta(1000, .x, .y))
      theta_est <- table(max.col(sim))
      theta_est <- theta_est / sum(theta_est)
      # Identify best arm
      a_t <- which.max(theta_est)
     
      a[t] <- a_t
      outcome <- rbinom(1, 1, prob=prob[a_t])
      alpha[a_t] <- alpha[a_t] + as.numeric(outcome == 1)
      beta[a_t] <- beta[a_t] + as.numeric(outcome == 0)
      regret[t] <- max(prob) - prob[a_t]
    }

    return(sum(regret))
  }

  # Only for measuring computation time
  time_finish_processing <- Sys.time()

  # Stop the cluster
  stopCluster(cl = cluster)

  # Keep track of the end time
  time_end <- Sys.time()
  
  print(time_end - time_start)
  
  return(mean(total_regret))
}
```

``` r
regret_val_ts <- c()
for (t in obs) {
  r <- ts_algorithm(context_prob, t)
  regret_val_ts <- c(regret_val_ts, r)
}
```

    ## Time difference of 5.744221 mins
    ## Time difference of 13.66547 mins
    ## Time difference of 17.55514 mins
    ## Time difference of 14.03017 mins
    ## Time difference of 18.5051 mins

``` r
regret_val_ts
```

    ## [1] 10.30 19.60 25.06 34.20 35.02

``` r
ggplot(data = tibble("t" = obs,
                     "ETC" = regret_val_etc,
                     "Epsilon Greedy" = regret_val_eg,
                     "UCB" = regret_val_ucb,
                     "MOSS" = regret_val_moss,
                     "TS" = regret_val_ts) %>% 
         pivot_longer(-t),
       aes(t, value, color=name)) +
  geom_line() +
  theme_classic() +
  labs(x = 'Number of survey respondents',
       y = 'Expected Cumulative Regret',
       color = 'Algorithm')
```

![](bandit_sim_v2_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
ggsave('algorithm_comparison.pdf')
```

    ## Saving 7 x 5 in image
