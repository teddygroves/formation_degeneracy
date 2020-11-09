data {
  int<lower=1> N_rxn;
  int<lower=1> N_cpd;
  int<lower=1> N_grp;
  matrix[N_cpd, N_rxn] S;
  matrix[N_cpd, N_grp] G;
  matrix[N_cpd, N_cpd] R;
  matrix[N_cpd, N_cpd] R_inv;
  matrix[N_grp, N_grp] RG;
  matrix[N_grp, N_grp] RG_inv;
  vector[N_rxn] y;
  vector<lower=0>[N_rxn] sigma;
  vector[N_grp] prior_gamma[2];
  vector[N_cpd] prior_theta[2];
  real prior_tau[2];
}
parameters {
  real<lower=0> tau;  // controls group additivity accuracy
  vector[N_cpd] eta_cpd;
  vector[N_grp] eta_grp;
}
transformed parameters {
  vector[N_cpd] theta = R_inv * eta_cpd;
  vector[N_grp] gamma = RG_inv * eta_grp;
}
model {
  target += normal_lpdf(theta | prior_theta[1], prior_theta[2]);
  target += normal_lpdf(gamma | prior_gamma[1], prior_gamma[2]);
  target += normal_lpdf(theta | G * gamma, tau);
  target += normal_lpdf(tau | prior_tau[1], prior_tau[2]);
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
