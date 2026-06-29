#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;

void ProgressBar(int cur, int total, int step = 10)
{
  if (cur % step != 0 && cur < total)
    return; // print every 'step' iters
  int console_width = 80;
  int bar_width = console_width - 20;

  int n_equals = std::min(static_cast<int>((static_cast<double>(cur) / total) * bar_width), bar_width);
  int n_space = bar_width - n_equals;

  static const char spinny[] = {'-', '\\', '|', '/'};
  char spinny_thing = spinny[n_equals % 4];

  std::string status = (cur >= total) ? "  Done   \n\n" : "  Loading";
  if (cur >= total)
    spinny_thing = '+';

  Rcpp::Rcout << "\r[" << std::string(n_equals, '=') << std::string(n_space, '-') << "]  "
              << spinny_thing << status << std::flush;
}


double cal_lik(const arma::vec& gammas,
               const arma::vec& betas,
               const arma::vec& Y,
               const arma::vec& constants,
               const std::vector<arma::mat>& K_cache,
               int L,
               double sigma2)
{

  const arma::uword n = Y.n_elem;

  if (gammas.n_elem != L || betas.n_elem != L) {
    Rcpp::stop("gammas, betas, and K_list must have the same length.");
  }
  if (sigma2 <= 0.0) {
    Rcpp::stop("sigma2 must be positive.");
  }

  arma::mat cov(n, n, fill::zeros);

  for (arma::uword l = 0; l < L; ++l) {
    const arma::mat& K = K_cache[l];

    if (K.n_rows != n || K.n_cols != n) {
      Rcpp::stop("Each matrix in K_list must be n x n.");
    }

    const double scale = gammas[l] * betas[l] * betas[l];
    if (scale != 0.0) {
      cov += scale * K;

    }
  }

  cov.diag() += sigma2;
  arma::mat cholcov;
  bool ok = arma::chol(cholcov, cov, "lower");

  if (!ok) {
    double jitter = 1e-10;
    for (int i = 0; i < 6 && !ok; ++i) {
      cov.diag() += jitter;
      ok = arma::chol(cholcov, cov, "lower");
      jitter *= 10.0;
    }
    if (!ok) {
      Rcpp::stop("Cholesky decomposition failed in cal_lik().");
    }
  }

  const double log_det_cov = 2.0 * sum(log(cholcov.diag()));
  const double log_const = -0.5 * static_cast<double>(n) * std::log(2.0 * M_PI) - 0.5 * log_det_cov;

  arma::vec resid = Y;

  const arma::vec z = solve(trimatl(cholcov), resid);
  const double quadratic_form = dot(z, z);

  return log_const - 0.5 * quadratic_form;
}

// The function below is for calculating the kernel function
// xmi - xm'i
mat sqdist_sym_1d(const vec& x){
  const int N = x.n_elem;
  mat D(N,N, arma::fill::zeros);

  arma::vec x2 = arma::square(x);
  D = arma::repmat(x2, 1, N) +
    arma::repmat(x2.t(), N, 1) -
    2.0 * (x * x.t());

  return D;
}

// we create a memory holder, if (xmi-xm'i) is not being compute, we compute, else we reuse
inline const mat& ensure_dsq(
    int j,
    const mat& X,
    std::vector<mat>& Dsq_cache,
    std::vector<unsigned char>& have_dsq
) {
  if (have_dsq[j]==0) {
    Dsq_cache[j] = sqdist_sym_1d(X.col(j));
    have_dsq[j] = 1;
  }
  return Dsq_cache[j];
}



arma::mat get_K_fast(int update,
                const arma::mat& X,
                const arma::vec& Eta,
                const arma::vec& lambda,
                const Rcpp::IntegerMatrix& idx,
                std::vector<arma::mat>& Dsq_cache,
                std::vector<unsigned char>& have_dsq,
                int eta = -1,
                const int l = -1)
{
  const int N = X.n_rows;
  arma::vec Eta_work = Eta;   // local copy
  if (update == 1 && l >= 0) {
    Eta_work(l) = eta;
  }

  if (arma::accu(Eta_work) == 0) {
    arma::mat out(N, N, arma::fill::zeros);
    out.diag().fill(1e-8);
    return out;
  }

  int n_const_interactions = 0;
  int n_const_main = 0;

  arma::mat G(N, N, arma::fill::zeros);
  arma::mat M(N, N, arma::fill::zeros);

  const double inv_lambda0 = 1.0 / lambda(0);
  const double inv_lambda1 = 1.0 / lambda(1);

  const int Ninteraction = idx.nrow();

  for (int k = 0; k < Ninteraction; ++k) {
    int a = idx(k, 0);
    int b = idx(k, 1);

    // if(Eta_work(a,b) == 1){
    if(Eta_work(k) == 1){
      if(a != b){
        const mat& Dsqa = ensure_dsq(a, X, Dsq_cache, have_dsq);
        const mat& Dsqb = ensure_dsq(b, X, Dsq_cache, have_dsq);
        G += arma::exp((-0.5 * inv_lambda0) * (Dsqa + Dsqb));
      }else{
        // calculate Xmi-Xm'i
        const mat& Dsq = ensure_dsq(a,X,Dsq_cache,have_dsq);
        M += arma::exp(-0.5 * inv_lambda1 * Dsq);
      }
    }

    // if non-important then exp(0)=1
    if(Eta_work(k) == 0){
      // G += 1.0;
      if(a != b){
        n_const_interactions+=1;
      }else{
        n_const_main +=1;
      }
    }

  }

  // add the zero interactions at very end
  if(n_const_interactions > 0){
    G +=n_const_interactions;
  }

  // add the zero interactions at very end
  if(n_const_main > 0){
    M +=n_const_main;
  }

  return G + M;

}

// [[Rcpp::export]]
arma::mat get_K_cpp(int update,
                    const arma::mat& X,
                    const arma::mat& Eta,
                    const arma::vec& lambda,
                    const Rcpp::IntegerMatrix& idx,
                    int eta = -1,
                    int i = -1,
                    int j = -1)
{
  const int N = X.n_rows;
  const int P = X.n_cols;

  arma::mat Eta_work = Eta;   // local copy

  if (update == 1 && i >= 0 && j >= 0) {
    Eta_work(j, i) = Eta_work(i, j) = eta;
  }

  if (arma::accu(Eta_work) == 0) {
    arma::mat out(N, N, arma::fill::zeros);
    out.diag().fill(1e-8);
    return out;
  }

  int n_const_interactions = 0;
  arma::mat G(N, N, arma::fill::zeros);

  // we need to change a new kernel for interaction
  const double inv_lambda0 = 1.0 / lambda(0);
  const int Nidx = idx.nrow();

  for (int k = 0; k < Nidx; ++k) {
    int i = idx(k, 0);
    int j = idx(k, 1);

    if(Eta_work(i,j) == 1 && i != j){

      // Lets use the kernel exp(D(xi,xi'))exp(D(xj,xj')) to capture the interaction
      arma::vec xj = X.col(j);
      arma::vec xj2 = arma::square(xj);
      arma::vec xi = X.col(i);
      arma::vec xi2 = arma::square(xi);

      mat D2i(N, N, arma::fill::zeros);
      mat D2j(N, N, arma::fill::zeros);

      for(int a = 0; a < N; ++a){
        for(int b = a; b < N; ++b){
          double di = xi[a] - xi[b];
          double dj = xj[a] - xj[b];

          D2i(a,b)=D2i(b,a) = di*di;
          D2j(a,b)=D2j(b,a) = dj*dj;
        }
      }

      G += arma::exp(-0.5 * inv_lambda0 * (D2i+D2j));
    }

    // if non-important then exp(0)=1
    if(Eta_work(i,j) == 0){
      // G += 1.0;
      ++n_const_interactions;
    }
  }

  // add the zero interactions at very end
  if(n_const_interactions > 0){
    G +=n_const_interactions;
  }

  const double inv_lambda1 = 1.0 / lambda(1);
  arma::vec eta_diag = Eta_work.diag();

  arma::uvec inactive = arma::find(eta_diag == 0.0);
  arma::uvec active   = arma::find(eta_diag != 0.0);

  arma::mat M(N, N, arma::fill::ones);
  M *= inactive.n_elem;

  for (arma::uword k = 0; k < active.n_elem; ++k) {
    int jj = active(k);

    arma::vec xj = X.col(jj);
    arma::vec xj2 = arma::square(xj);

    arma::mat D2 =
      arma::repmat(xj2, 1, N) +
      arma::repmat(xj2.t(), N, 1) -
      2.0 * (xj * xj.t());

    M += arma::exp(-0.5 * inv_lambda1 * eta_diag(jj) * D2);

  }

  return G + M;

}


void update_eta(int k,
               int l,
                int L,
                double sigma2,
                const arma::vec& lambdas,
                double& ll,
                const arma::vec& gammas,
                const arma::vec& betas,
                const arma::vec& constants,
                const arma::vec& Y,
                arma::vec& Eta,
                const arma::mat& X,
                std::vector<arma::mat>& K_cache,
                const Rcpp::IntegerMatrix& idx,
                std::vector<arma::mat>& Dsq_cache,
                std::vector<unsigned char>& have_dsq)
{
  const arma::vec lambda = lambdas.subvec(2 * l, 2 * l + 1);

  const double prob_eta = constants(0);
  const double eta_curr = Eta(k);

  const double log_lik1 = ll;
  const double log_prior1 =
    eta_curr * std::log(prob_eta) + (1.0 - eta_curr) * std::log(1.0 - prob_eta);
  const double p1 = log_prior1 + log_lik1;

  const double eta_star = 1.0 - eta_curr;

  const arma::mat K_old = K_cache[l];

  // if Eta is a adjacency matrix then arma::accu(Eta) < 4.0
  // Else we do
  if(arma::accu(Eta) < 2.0){
    const arma::mat K_star = get_K_fast(1, X, Eta, lambda, idx,Dsq_cache,have_dsq,eta_star, k);
    K_cache[l] = K_star;
  }else{
    // Let's not recompute the whole get_K_fast
    arma::mat K_star = K_old;
    int i = idx(k, 0);
    int j = idx(k, 1);
    if(i == j){
      const double inv_lambda = 1.0 / lambda(1);
      const arma::mat& Dsq = ensure_dsq(i,X,Dsq_cache,have_dsq);
      K_star -= arma::exp(-0.5*inv_lambda*eta_curr*Dsq);
      K_star += arma::exp(-0.5*inv_lambda*eta_star*Dsq);
    }else{
      const double inv_lambda = 1.0 / lambda(0);
      const arma::mat Dsq = ensure_dsq(i,X,Dsq_cache,have_dsq) +
        ensure_dsq(j,X,Dsq_cache,have_dsq);
      K_star -= arma::exp(-0.5*inv_lambda*eta_curr*Dsq);
      K_star += arma::exp(-0.5*inv_lambda*eta_star*Dsq);
    }
    K_cache[l] = K_star;
  }


  const double log_lik2 = cal_lik(gammas, betas, Y, constants, K_cache, L, sigma2);

  const double log_prior2 =
    eta_star * std::log(prob_eta) + (1.0 - eta_star) * std::log(1.0 - prob_eta);
  const double p2 = log_lik2 + log_prior2;

  const double m = std::max(p1, p2);
  const double w1 = std::exp(p1 - m);
  const double w2 = std::exp(p2 - m);

  double prob_one;
  if (eta_curr == 1.0) {
    prob_one = w1 / (w1 + w2);
    } else {
    prob_one = w2 / (w1 + w2);
  }

  const double eta_new = (arma::randu() < prob_one) ? 1.0 : 0.0;

  if (eta_new != eta_curr) {
    Eta(k) = eta_new;
    ll = log_lik2;
  } else {
    K_cache[l] = K_old;  // revert
  }
}

double prior_beta(const vec& constants,
                  double gamma,
                  double beta)
{
  double tau2 = (gamma == 1) ? constants(3) : constants(2);
  return -0.5*std::log(tau2) - 0.5*(beta*beta)/tau2;
}

void update_beta(double& ll,
                 double sigma2,
                 int L,
                 int l,
                 double& num_accept,
                 const vec& gammas,
                 vec& betas,
                 const vec& Y,
                 const vec& constants,
                 const vec& cov_beta,
                 const std::vector<arma::mat>& K_cache) {

  const double prop_var_slab = std::exp(cov_beta(l));
  const double beta = betas(l);
  const int gamma = gammas(l);

  double beta_star = beta;
  beta_star += (gamma == 1) ? prop_var_slab * randn() : 0.01 * randn();

  const double prior_star = prior_beta(constants, gamma, beta_star);
  const double prior_curr = prior_beta(constants, gamma, beta);

  vec betas_star = betas;
  betas_star(l) = beta_star;

  const double ll_star = cal_lik(gammas, betas_star, Y, constants, K_cache, L, sigma2);
  const double log_acc = (prior_star - prior_curr) + (ll_star - ll);

  if (std::log(randu()) < log_acc) {
    ll = ll_star;
    betas = betas_star;
    num_accept++;
  }
}

void update_gamma(int l,
                  int L,
                  double sigma2,
                  double& ll,
                  arma::vec& gammas,
                  const arma::vec& betas,
                  const arma::vec& Y,
                  const arma::vec& constants,
                  const std::vector<arma::mat>& K_cache)
{
  const double prob_gamma = constants(1);
  const double gamma_curr = gammas(l);
  const double beta_l = betas(l);

  // current state
  const double log_lik1 = ll;
  const double log_prior1 = prior_beta(constants, gamma_curr, beta_l);
  const double p1 =
    log_lik1 + log_prior1 +
    ((1.0 - gamma_curr) * std::log(1.0 - prob_gamma) +
    gamma_curr * std::log(prob_gamma));

  // flipped state: temporarily flip gamma in-place
  const double gamma_star = 1.0 - gamma_curr;
  gammas(l) = gamma_star;

  const double log_lik2 = cal_lik(gammas, betas, Y, constants, K_cache, L, sigma2);
  const double log_prior2 = prior_beta(constants, gamma_star, beta_l);
  const double p2 =
    log_lik2 + log_prior2 +
    ((1.0 - gamma_star) * std::log(1.0 - prob_gamma) +
    gamma_star * std::log(prob_gamma));

  // stabilized probability
  const double m = std::max(p1, p2);
  const double w1 = std::exp(p1 - m);
  const double w2 = std::exp(p2 - m);
  const double prob_flip = w2 / (w1 + w2);

  const bool do_flip = (arma::randu() < prob_flip);

  if (do_flip) {
    ll = log_lik2;   // keep flipped gamma already stored in gammas(l)
  } else{
    gammas(l) = gamma_curr;  // revert
  }
}

rowvec mat_to_vec(mat C)
{
  int n = C.n_cols;
  rowvec X(n * n);

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      X(i * n + j) = C(i, j); // Using your specific indexing logic
    }
  }
  return X;
}

// maybe we should change this into log normal
double prior_lambda(vec constants, double lambda)
{
  double a = constants(4);
  double b = constants(5);

  // prior for sigma
  double log_prior = a * log(b) - lgamma(a) - (a + 1.0) * log(lambda) - b / lambda;
  return log_prior;
}

void update_lambda(vec& lambdas,
                   double sigma2,
                   double& ll,
                   const vec& Y,
                   const vec& constants,
                   const vec& gammas,
                   const vec& betas,
                   const mat& X,
                   const vec& Eta_l,
                   std::vector<arma::mat>& K_cache,
                   int L,
                   int l,
                   const vec& proposal_sds,
                   vec& accept_lambdas,
                   int iteration,
                   int burnin,
                   const Rcpp::IntegerMatrix& idx,
                   std::vector<arma::mat>& Dsq_cache,
                   std::vector<unsigned char>& have_dsq)
{
  const double proposal_sd0 = proposal_sds[2 * l];
  const double proposal_sd1 = proposal_sds[2 * l + 1];

  double lambda0 = lambdas[2 * l];
  double lambda1 = lambdas[2 * l + 1];

  arma::vec lambda_star(2);

  // ----- update interaction lambda -----
  double log_lambda_curr = std::log(lambda0);
  double log_lambda_prop = log_lambda_curr + std::exp(proposal_sd0) * randn();

  lambda_star[0] = std::exp(log_lambda_prop);
  lambda_star[1] = lambda1;

  const arma::mat K_old = K_cache[l];
  const arma::mat K_star0 = get_K_fast(0, X, Eta_l, lambda_star, idx,Dsq_cache,have_dsq);

  K_cache[l] = K_star0;
  double ll_star = cal_lik(gammas, betas, Y, constants, K_cache, L, sigma2);

  double logprior_curr = -0.5 * std::pow(log_lambda_curr, 2) / 0.25;
  double logprior_star = -0.5 * std::pow(log_lambda_prop, 2) / 0.25;
  double log_acc = (ll_star - ll) + (logprior_star - logprior_curr);

  if (std::log(randu()) < log_acc) {
    lambdas[2 * l] = lambda_star[0];
    ll = ll_star;
    accept_lambdas[2 * l] += 1.0;
    lambda0 = lambdas[2 * l];
  } else {
    K_cache[l] = K_old;
  }

  // ----- update main-effect lambda -----
  log_lambda_curr = std::log(lambda1);
  log_lambda_prop = log_lambda_curr + std::exp(proposal_sd1) * randn();

  lambda_star[0] = lambda0;
  lambda_star[1] = std::exp(log_lambda_prop);

  const arma::mat K_old2 = K_cache[l];
  const arma::mat K_star1 = get_K_fast(0, X, Eta_l, lambda_star, idx,Dsq_cache,have_dsq);

  K_cache[l] = K_star1;
  ll_star = cal_lik(gammas, betas, Y, constants, K_cache, L, sigma2);

  logprior_curr = -0.5 * std::pow(log_lambda_curr, 2) / 0.25;
  logprior_star = -0.5 * std::pow(log_lambda_prop, 2) / 0.25;
  log_acc = (ll_star - ll) + (logprior_star - logprior_curr);

  if (std::log(randu()) < log_acc) {
    lambdas[2 * l + 1] = lambda_star[1];
    ll = ll_star;
    accept_lambdas[2 * l + 1] += 1.0;
  } else {
    K_cache[l] = K_old2;
  }
}

double prior_sigma(vec constants,double sigma2){
  // prior for sigma
  double log_prior = constants(4)*log(constants(5))-lgamma(constants(4))-(constants(4) + 1.0)*log(sigma2)-constants(5) / sigma2;
  return log_prior;
}

void update_variance(double& sigma2,
                     const vec& Y,
                     const vec& constants,
                     const vec& gammas,
                     const vec& betas,
                     const std::vector<arma::mat>& K_cache,
                     int L,
                     double& ll,
                     double proposal_sd,
                     int& accept_sigma2) {

  double log_sigma2_curr = std::log(sigma2);
  double log_sigma2_prop = log_sigma2_curr + exp(proposal_sd)*randn();
  double sigma2_star = exp(log_sigma2_prop);

  double ll_star = cal_lik(gammas,betas,Y,constants,K_cache,L,sigma2_star);
  double logprior_curr = prior_sigma(constants,sigma2);
  double logprior_star = prior_sigma(constants,sigma2_star);
  double log_jacobian = log_sigma2_prop - log_sigma2_curr;

  double log_acc = (ll_star - ll) + (logprior_star - logprior_curr) + log_jacobian;

  if (log(randu()) < log_acc) {
    ll = ll_star;
    sigma2 = sigma2_star;
    accept_sigma2++;
  }
}

// [[Rcpp::export]]
IntegerMatrix extract_edges_fast(const arma::mat& A) {

  arma::uword p = A.n_rows;

  // upper triangle (including diagonal)
  arma::uvec lin_idx = arma::find( arma::trimatu(A, 0) > 0 );

  int n_edges = (int)lin_idx.n_elem;
  IntegerMatrix out(n_edges, 2);
  std::vector<bool> seen(p, false);

  // Off-diagonal edges
  for (arma::uword k = 0; k < lin_idx.n_elem; ++k) {
    arma::uword lin = lin_idx(k);

    arma::uword r = lin % p;   // row
    arma::uword c = lin / p;   // col

    out((int)k, 0) = (int)r;   // 0-based
    out((int)k, 1) = (int)c;   // 0-based

    if (!seen[r]) {
      seen[r] = true;
    }
    if (!seen[c]) {
      seen[c] = true;
    }

  }
  return out;
}


vec extract_eta(const IntegerMatrix& idx,
                       const mat& Eta) {

  const int Nidx = idx.nrow();
  vec Eta_sub(Nidx, fill::zeros);

  for (int k = 0; k < Nidx; ++k) {
    int i = idx(k, 0);
    int j = idx(k, 1);
    Eta_sub(k) = Eta(i,j);
  }

  return Eta_sub;
}


// [[Rcpp::export]]
Rcpp::List run_mcmc(int iterations, int burn, int L,
                    const Rcpp::List& As,
                    const arma::vec& Y,
                    const arma::mat& X,
                    const Rcpp::List& inits)
{
  const int n = Y.n_elem;
  const int p = X.n_cols;

  arma::vec constants = inits[0];
  double sigma2 = inits[1];
  arma::vec lambdas = inits[2];
  arma::vec betas = inits[3];
  arma::vec gammas = inits[4];
  Rcpp::List Etas_in = inits[5];

  // setup memory storage for distance matrix
  std::vector<mat> Dsq_cache(p);
  std::vector<unsigned char> have_dsq(p, 0);

  // here we need to change list into C++ vectors
  std::vector<arma::mat> A_cache(L), K_cache(L);
  std::vector<arma::vec> Eta_cache(L);
  std::vector<Rcpp::IntegerMatrix> idx_cache(L);
  std::vector<arma::mat> eta_store(L);

  // we are creating each matrix inside the vector
  for (int l = 0; l < L; ++l) {
    A_cache[l] = Rcpp::as<arma::mat>(As[l]);
    idx_cache[l] = extract_edges_fast(A_cache[l]);
    Eta_cache[l] = extract_eta(idx_cache[l] ,Rcpp::as<arma::mat>(Etas_in[l]) % A_cache[l]);

    K_cache[l] = get_K_fast(0, X, Eta_cache[l],
                           lambdas.subvec(2*l, 2*l + 1), idx_cache[l],Dsq_cache,have_dsq);

    eta_store[l].set_size(iterations, idx_cache[l].nrow());
    eta_store[l].fill(arma::datum::nan);
  }

  arma::mat Y_whole(n, iterations);
  arma::mat beta_save(L, iterations);
  arma::mat gamma_save(L, iterations);
  arma::mat lambda_save(2 * L, iterations);
  arma::vec sigma_save(iterations);
  arma::vec ll_save(iterations);

  int accept_sigma2 = 0;
  int total_sigma2 = 0;
  double proposal_sig = -1.0;
  arma::vec accept_beta(L, arma::fill::zeros);
  arma::vec total_accept(L, arma::fill::zeros);
  arma::vec cov_beta(L, arma::fill::value(-1.0));
  arma::vec proposal_sd(2 * L, arma::fill::value(-1.0));
  arma::vec accept_lambda(2 * L, arma::fill::zeros);
  arma::vec total_accept_lambda(2 * L, arma::fill::zeros);


  double ll = cal_lik(gammas, betas, Y, constants, K_cache, L, sigma2);

  for (int iteration = 1; iteration <= iterations; ++iteration) {
    // Rcout << iteration << "\n" << endl;

    ProgressBar(iteration, iterations, 50);

    for (int l = 0; l < L; ++l) {
      arma::vec& Eta = Eta_cache[l];

      Rcpp::IntegerMatrix& idx = idx_cache[l];

      update_beta(ll, sigma2, L, l, accept_beta[l], gammas, betas, Y, constants, cov_beta, K_cache);

      update_gamma(l, L, sigma2, ll, gammas, betas, Y, constants, K_cache);

      const int Nidx = idx.nrow();
      for (int k = 0; k < Nidx; ++k) {
        update_eta(k, l, L, sigma2, lambdas, ll, gammas, betas,
                   constants, Y, Eta, X, K_cache, idx,Dsq_cache,have_dsq);
      }

      update_lambda(lambdas, sigma2, ll, Y, constants, gammas, betas,
                    X, Eta, K_cache, L, l, proposal_sd, accept_lambda,
                    iteration, burn, idx,Dsq_cache,have_dsq);

      eta_store[l].row(iteration - 1) = Eta.t();

      if (iteration % 100 == 0) {
        if (iteration <= burn) {
          proposal_sd[2*l] += (accept_lambda[2*l] / 100.0 > 0.4) ? 0.1 : -0.1;
          proposal_sd[2*l + 1] += (accept_lambda[2*l + 1] / 100.0 > 0.4) ? 0.1 : -0.1;
          cov_beta[l] += (accept_beta[l] / 100.0 > 0.4) ? 0.1 : -0.1;
        } else {
          total_accept_lambda[2*l] += accept_lambda[2*l];
          total_accept_lambda[2*l + 1] += accept_lambda[2*l + 1];
          total_accept[l] += accept_beta[l];
        }
        accept_lambda[2*l] = 0;
        accept_lambda[2*l + 1] = 0;
        accept_beta[l] = 0;
      }
    }

    update_variance(sigma2, Y, constants, gammas, betas, K_cache, L, ll, proposal_sig, accept_sigma2);

    if (iteration % 100 == 0) {
      double accept_100 = accept_sigma2;
      if (iteration <= burn) {
        proposal_sig += (accept_100 / 100.0 > 0.4) ? 0.1 : -0.1;
      } else {
        total_sigma2 += accept_sigma2;
      }
      accept_sigma2 = 0;
    }

    beta_save.col(iteration - 1) = betas;
    gamma_save.col(iteration - 1) = gammas;
    lambda_save.col(iteration - 1) = lambdas;
    sigma_save(iteration - 1) = sigma2;
    ll_save(iteration - 1) = ll;
  }

  // For output purpose, we want to change the format of Eta to a list
  Rcpp::List eta_whole_out(L);
  Rcpp::List idx_out(L);

  for (int l = 0; l < L; ++l){
    eta_whole_out[l] = eta_store[l];
    idx_out[l] = idx_cache[l] + 1;
  }

  return Rcpp::List::create(
    Rcpp::_["gamma_save"] = gamma_save.t(),
    Rcpp::_["beta_save"] = beta_save.t(),
    Rcpp::_["sigma_save"] = sigma_save,
    Rcpp::_["eta_whole"] = eta_whole_out,
    Rcpp::_["eta_idx"] = idx_out,
    Rcpp::_["lambda_save"] = lambda_save.t(),
    Rcpp::_["ll_save"] = ll_save
  );

}
