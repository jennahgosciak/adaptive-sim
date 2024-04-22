Bandit Sim
================

# Explore-then-commit Algorithm

``` r
## explore then commit algorithm
etc_algorithm <- function(prob, n, m=NULL,
                          reps=1000, k=4) {
  
  total_regret <- c()
  for (r in 1:reps) {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    
    for (t in 1:(m*k)) {
      a_t <- t%%k + 1
      a[t] <- a_t
      mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
      regret[t] <- max(prob) - prob[a_t]
    }
    
    mu_mean <- 
      apply(mu[1:(m*k),], 2, 
            mean, na.rm = TRUE)
    for (t in ((m*k)+1):n) {
          a_t <- which.max(mu_mean)
          a[t] <- a_t
          regret[t] <- max(prob) - prob[a_t]
    }
    total_regret <- c(total_regret, sum(regret))
  }
  return(mean(total_regret))
}
```

``` r
regret_val <- c()
m_vals <- c(1, 2, 5, 10, 20, 30, 40)
for (m in m_vals) {
  r <- etc_algorithm(c(0.6, 0.5, 0.3, 0.7), 1000, m = m)
  regret_val <- c(regret_val,
                  r)
}


ggplot(data = data.frame("m" = m_vals,
                         "expected regret" = regret_val),
       aes(m, `expected.regret`)) +
  geom_point() +
  theme_classic()
```

![](bandit_sim_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
regret_val_etc <- c()
for (t in seq(100, 1000, 100)) {
  r <- etc_algorithm(c(0.6, 0.5, 0.3, 0.7), t, m = 2)
  regret_val_etc <- c(regret_val_etc,
                  r)
}

ggplot(data = tibble("t" = seq(100, 1000, 100),
                         "expected regret (ETC)" = regret_val_etc),
       aes(t, `expected regret (ETC)`)) +
  geom_point() +
  theme_classic()
```

![](bandit_sim_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Epsilon Greedy Algorithm

``` r
# epsilon-greedy algorithm
epsilon_greedy_algorithm <- function(prob, n, eps,
                          reps=100, k=4) {
  total_regret <- c()
  for (r in 1:reps) {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    
    for (t in 1:n) {
      
      if ((runif(1) > eps) & (t > 2)) {
        mu_mean <- 
          apply(mu[1:(t-1),], 2, 
                mean, na.rm = TRUE)
        a_t <- which.max(mu_mean)
      } else {
        a_t <- sample(c(1,2), 1)
      }
        a[t] <- a_t
        mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
        regret[t] <- max(prob) - prob[a_t]
    }
    
    total_regret <- c(total_regret, sum(regret))
  }
  return(mean(total_regret))
}
```

``` r
regret_val_eg <- c()
for (t in seq(100, 1000, 100)) {
  r <- epsilon_greedy_algorithm(c(0.6, 0.5, 0.3, 0.7), t, 
                                eps=0.1)
  regret_val_eg <- c(regret_val_eg,
                  r)
}

ggplot(data = tibble("t" = seq(100, 1000, 100),
                     "expected regret (ETC)" = regret_val_etc,
                         "expected regret (epsilon greedy)" = regret_val_eg) %>% 
         pivot_longer(-t),
       aes(t, value, color=name)) +
  geom_line() +
  theme_classic() +
  labs(x = 't',
       y = 'Expected Regret')
```

![](bandit_sim_files/figure-gfm/unnamed-chunk-6-1.png)<!-- --> \## UCB

``` r
## upper confidence bound
ucb_algorithm <- function(prob, n, delta,
                          reps=100, k=4) {
  total_regret <- c()
  for (r in 1:reps) {
    regret <- vector("numeric", n)
    a <- vector("integer", n)
    mu <- matrix(nrow=n, ncol=k)
    pulls <- vector("integer", k)
    ucb <- matrix(nrow=n, ncol=k)
    
    for (t in 1:n) {
      
      if (t > 2) {
        mu_mean <- 
          apply(mu[1:(t-1),], 2, 
                mean, na.rm = TRUE)
        ucb <- mu_mean + sqrt((2 * log(1 / delta)) / pulls)
        a_t <- which.max(ucb)
      } else {
        a_t <- sample(c(1,2), 1)
      }
        a[t] <- a_t
        pulls[a_t] <- pulls[a_t] + 1
        mu[t, a_t] <- rbinom(1, 1, prob=prob[a_t])
        regret[t] <- max(prob) - prob[a_t]
    }
    
    total_regret <- c(total_regret, sum(regret))
  }
  return(mean(total_regret))
}
```

``` r
regret_val_ucb <- c()
for (t in seq(100, 1000, 100)) {
  r <- ucb_algorithm(c(0.6, 0.5, 0.3, 0.7), t, 
                                delta=1/(t^2))
  regret_val_ucb <- c(regret_val_ucb,
                  r)
}
```

``` r
ggplot(data = tibble("t" = seq(100, 1000, 100),
                     "ETC" = regret_val_etc,
                         "Epsilon Greedy" = regret_val_eg,
                     "UCB" = regret_val_ucb) %>% 
         pivot_longer(-t),
       aes(t, value, color=name)) +
  geom_line() +
  theme_classic() +
  labs(x = 't',
       y = 'Expected Regret')
```

![](bandit_sim_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->