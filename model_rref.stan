data {
  int<lower=1> N_rxn;
  int<lower=1> N_cpd;
  matrix[N_cpd, N_rxn] S;
  matrix[N_cpd, N_cpd] R;  // matrix defining a reparameterisation
  vector[N_rxn] y;
  vector<lower=0>[N_rxn] error_scale;
  vector[N_cpd] prior_loc_theta;
  vector<lower=0>[N_cpd] prior_scale_theta;
}
parameters {
  vector[N_cpd] gamma;
}
transformed parameters {
  vector[N_cpd] theta = R \ gamma;
}
model {
  target += normal_lpdf(theta | prior_loc_theta, prior_scale_theta);
  // no jacobian as gamma -> theta is a linear transformation 
  target += normal_lpdf(y | S' * theta, error_scale);
}
generated quantities {
  vector[N_rxn] yrep;
  vector[N_rxn] log_lik;
  {
    vector[N_rxn] yhat = S' * theta;
    for (n in 1:N_rxn){
      yrep[n] = normal_rng(yhat[n], error_scale[n]);
      log_lik[n] = normal_lpdf(y[n] | yhat[n], error_scale[n]);
    }
  }                             
}
