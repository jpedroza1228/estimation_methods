data {
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> C;
  int<lower=1> K;
  matrix<lower=0,upper=1> [J,I] Y;
  matrix<lower=0,upper=1> [I,K] Q;
  matrix<lower=0,upper=1> [C,K] alpha;
  matrix<lower=0,upper=1> [I,C] xi;
}

parameters {
  simplex[C] nu;
  vector<lower=0, upper=1>[I] slip;
  vector<lower=0, upper=1>[I] guess;
}

transformed parameters {
  vector[C] log_nu;
  matrix[I,C] pi;

  log_nu = log(nu);

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow((1 - slip[i]), xi[i,c]) *
      pow(guess[i], (1 - xi[i,c]));
    }
  }
}

model{
  array[C] real ps;
  array[I] real eta;

  slip[1] ~ beta(2, 28);
  slip[2] ~ beta(2, 28);
  slip[3] ~ beta(2, 28);
  slip[4] ~ beta(4, 26);
  slip[5] ~ beta(4, 26);
  slip[6] ~ beta(4, 26); 
  slip[7] ~ beta(4, 26);
  slip[8] ~ beta(4, 26); 
  slip[9] ~ beta(4, 26); 
  slip[10] ~ beta(4, 26); 
  slip[11] ~ beta(4, 26);
  slip[12] ~ beta(4, 26);
  slip[13] ~ beta(4, 26);
  slip[14] ~ beta(10, 20);
  slip[15] ~ beta(10, 20);
  slip[16] ~ beta(10, 20); 
  slip[17] ~ beta(10, 20);
  slip[18] ~ beta(10, 20); 
  slip[19] ~ beta(10, 20); 
  slip[20] ~ beta(10, 20);
  slip[21] ~ beta(10, 20);
  slip[22] ~ beta(10, 20);
  slip[23] ~ beta(10, 20);
  slip[24] ~ beta(10, 20);
  slip[25] ~ beta(10, 20);
  slip[26] ~ beta(10, 20); 
  slip[27] ~ beta(10, 20);
  slip[28] ~ beta(10, 20); 
  slip[29] ~ beta(10, 20); 
  slip[30] ~ beta(10, 20); 
  slip[31] ~ beta(10, 20);
  slip[32] ~ beta(10, 20);
  slip[33] ~ beta(10, 20);
  slip[34] ~ beta(10, 20);

  for (i in 1:I){
    guess[i] ~ beta(6, 24);
  }

  for (j in 1:J){
    for (c in 1:C){
      for (i in 1:I){
        real p = fmin(fmax(pi[i,c], 1e-9), (1 - 1e-9));
        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
      }
      ps[c] = log_nu[c] + sum(eta);
    }
    target += log_sum_exp(ps);
  }
}

generated quantities {
  matrix[J,C] prob_resp_class;
  matrix[J,K] prob_resp_attr;
  array[I] real eta;
  row_vector[C] prob_joint;
  vector[J] log_lik;
  array[C] real prob_attr_class;
  matrix[J,I] y_rep;

  for (j in 1:J){
   for (c in 1:C){
     for(i in 1:I){
       real p = fmin(fmax(pi[i,c], 1e-9), (1 - 1e-9));
       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
     }
     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));
   }
   log_lik[j] = log_sum_exp(prob_joint);
   prob_resp_class[j] = prob_joint/sum(prob_joint);
  }

  for (j in 1:J){
    for (k in 1:K){
      for (c in 1:C){
        prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];
      }
      prob_resp_attr[j,k] = sum(prob_attr_class);
    }
  }

  for (j in 1:J){
    int z = categorical_rng(nu);
    for (i in 1:I){
      y_rep[j,i] = bernoulli_rng(pi[i,z]);
    }
  }
}