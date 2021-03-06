data {
  int<lower=1> N_rxn;
  int<lower=1> N_cpd;
  int<lower=1> N_grp;
  matrix[N_cpd, N_rxn] S;
  matrix[N_cpd, N_grp] G;
  vector[N_rxn] y;
  vector<lower=0>[N_rxn] sigma;
  vector[N_cpd] prior_theta[2];
  vector[N_grp] prior_gamma[2];
  real prior_tau[2];
}
parameters {
  real<lower=0> tau;
  vector[N_cpd] theta;
  vector[N_grp] gamma;
}
model {
  target += normal_lpdf(theta | prior_theta[1], prior_theta[2]);
  target += normal_lpdf(gamma | prior_gamma[1], prior_gamma[2]);
  target += normal_lpdf(tau | prior_tau[1], prior_tau[2]);
  target += normal_lpdf(theta | G * gamma, tau);  // approximate group additivity
  target += normal_lpdf(y | S' * theta, sigma);
}
generated quantities {
  vector[N_rxn] yrep;
  real log_lik = 0;
  {
    vector[N_rxn] yhat = S' * theta;
    for (n in 1:N_rxn){
      yrep[n] = normal_rng(yhat[n], sigma[n]);
      log_lik += normal_lpdf(y[n] | yhat[n], sigma[n]);
    }
  }                             
}
