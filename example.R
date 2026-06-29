set.seed(1)
N <- 50
burnin <- 10
L <- 3
n <- 100
p <- 50
sigma2 <- 1
beta <- c(0, 2, 2)
gamma <- ifelse(beta == 0, 0, 1)

pathway <- simulate_pathway(
  p = p,
  n_pathways = L,
  pathway_p = 10,
  n_edges = 15,
  n_imp_edges = 5,
  n_imp_main = 2,
  main = TRUE,
  seed = 1
)

X <- runif(n*p, min = 0, max = 1)
X <- matrix(X, nrow = n,ncol = p)
X <- scale(X)

A <- pathway$A
A_imp <- pathway$A_imp

FX <- matrix(0, nrow(X), L)

for(l in 1:L){
  K <- get_K(X, A_imp[[l]], c(1, 1))
  # Sample fx from MVN(0, K).
  fx <- drop(t(chol(K + 1e-8 * diag(nrow(X)))) %*% rnorm(nrow(X)))
  # store results
  FX[, l] <- fx
}

Y <- drop(FX %*% beta + rnorm(n, sd = sqrt(sigma2)))

result <- bps_mkl(
  iterations = N,
  burn = burnin,
  L = L,
  As = A,
  Y = Y,
  X = X,
  sigma2 = sigma2,
  betas = beta,
  gammas = gamma,
  Etas = A_imp
)

# # Nyström approximate
# result1 <- bps_mkl_app(
#   iterations = N,
#   burn = burnin,
#   L = L,
#   As = A,
#   Y = Y,
#   X = X,
#   m = 50,
#   sigma2 = sigma2,
#   betas = beta,
#   gammas = gamma,
#   Etas = A_imp
# )

res_summary(result,Figure=TRUE)
