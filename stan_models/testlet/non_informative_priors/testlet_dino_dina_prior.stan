// TODO include testlet effect (follow testlet_dino_dina_model.stan syntax from slightly_informative_priors)

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

  for (i in 1:I){
    slip[i] ~ beta(1, 1);
    guess[i] ~ beta(1, 1);
  }
}

generated quantities {
  matrix[J,I] y_rep;

  for (j in 1:J){
    int z = categorical_rng(nu);
    for (i in 1:I){
      y_rep[j,i] = bernoulli_rng(pi[i,z]);
    }
  }
}