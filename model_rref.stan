data {
  int<lower=1> N_rxn;
  int<lower=1> N_cpd;
  matrix[N_cpd, N_rxn] S;
  matrix[N_rxn, N_cpd] S_T_rref;
  vector[N_rxn] y;
  vector<lower=0>[N_rxn] error_scale;
  vector[N_cpd] prior_loc_theta;
  vector<lower=0>[N_cpd] prior_scale_theta;
}
transformed data {
  vector[N_rxn] prior_loc_theta_free = S_T_rref * prior_loc_theta;
  vector[N_rxn] prior_scale_theta_free = fabs(S_T_rref) * prior_scale_theta;
}
parameters {
  vector[N_rxn] theta_free;
}
model {
  vector[N_rxn] yhat = S' * (theta_free' * S_T_rref)';
  target += normal_lpdf(theta_free | prior_loc_theta_free, prior_scale_theta_free);
  target += normal_lpdf(y | yhat, error_scale);
}
generated quantities {
  vector[N_rxn] yrep;
  {
    vector[N_rxn] yhat = S' * (theta_free' * S_T_rref)';
    for (n in 1:N_rxn){
      yrep[n] = normal_rng(yhat[n], error_scale[n]);
    }
  }
}
