#' Run BPS-MKL from individual initial values
#'
#' `bsp_mkl()` collects the individual initial values into the list expected by
#' the C++ sampler and then runs the dense-kernel BPS-MKL MCMC algorithm.
#'
#' @param iterations Integer. Total number of MCMC iterations to run.
#' @param burn Integer. Number of burn-in iterations used for proposal
#'   adaptation.
#' @param L Integer. Number of pathways.
#' @param As List of `L` numeric adjacency matrices. Each matrix defines the
#'   candidate main effects and interactions for one pathway.
#' @param Y Numeric response vector or `n x 1` matrix.
#' @param X Numeric `n x p` design matrix, with observations in rows and
#'   predictors in columns.
#' @param P_eta Numeric scalar eta inclusion probability.
#' @param P_gamma Numeric scalar gamma inclusion probability.
#' @param sigma2 Numeric scalar initial residual variance.
#' @param lambdas Numeric vector of length `2 * L`, with interaction and
#'   main-effect bandwidths for each pathway.
#' @param betas Numeric vector of length `L` containing initial beta values.
#' @param gammas Numeric vector of length `L` containing initial gamma values.
#' @param Etas List of `L` numeric eta matrices.
#'
#' @return A list with MCMC draws and diagnostics:
#' \describe{
#'   \item{`gamma_save`}{An `iterations` by `L` matrix of gamma draws.}
#'   \item{`beta_save`}{An `iterations` by `L` matrix of beta draws.}
#'   \item{`sigma_save`}{A numeric vector of sigma2 draws.}
#'   \item{`eta_whole`}{A list of `L` matrices storing eta draws for the
#'   candidate terms in each pathway.}
#'   \item{`eta_idx`}{A list of `L` two-column matrices giving the one-based
#'   predictor indices corresponding to the columns of `eta_whole`.}
#'   \item{`lambda_save`}{An `iterations` by `2 * L` matrix of lambda draws.}
#'   \item{`ll_save`}{A numeric vector of log-likelihood values.}
#'   \item{`time`}{Time used to run the algorithm, returned
#'   as a `difftime` object.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' L <- 3
#' n <- 100
#' p <- 50
#' sigma2 <- 1
#' beta <- c(0, 2, 2)
#' gamma <- ifelse(beta == 0, 0, 1)
#'
#' pathway <- simulate_pathway(
#'   p = p,
#'   n_pathways = L,
#'   pathway_p = 10,
#'   n_edges = 15,
#'   n_imp_edges = 5,
#'   n_imp_main = 2,
#'   main = TRUE,
#'   seed = 1
#' )
#'
#' X <- matrix(runif(n * p), nrow = n, ncol = p)
#' X <- scale(X)
#'
#' A <- pathway$A
#' A_imp <- pathway$A_imp
#' FX <- matrix(0, nrow = n, ncol = L)
#'
#' for (l in seq_len(L)) {
#'   K <- get_K(X, A_imp[[l]], c(1, 1))
#'   # Sample fx from MVN(0, K).
#'   FX[, l] <- drop(t(chol(K + 1e-8 * diag(n))) %*% rnorm(n))
#' }
#'
#' Y <- drop(FX %*% beta + rnorm(n, sd = sqrt(sigma2)))
#'
#' result <- bsp_mkl(
#'   iterations = 5000,
#'   burn = 1000,
#'   L = L,
#'   As = A,
#'   Y = Y,
#'   X = X,
#'   sigma2 = sigma2,
#'   betas = beta,
#'   gammas = gamma,
#'   Etas = A_imp
#' )
#' }
#'
#' @export
bsp_mkl <- function(iterations, burn, L, As, Y, X,
                    P_eta = 0.2,
                    P_gamma = 0.2,
                    sigma2 = 0.1,
                    lambdas = NULL,
                    betas = NULL,
                    gammas = NULL,
                    Etas = NULL) {
  if (nrow(X) != length(Y)) {
    stop("Y must have length nrow(X).", call. = FALSE)
  }

  p <- ncol(X)
  if (length(As) != L) {
    stop("As must be a list of length L.", call. = FALSE)
  }

  # p_eta, p_gamma, spike beta, slab beta, sigma a, sigma b
  constants <- c(P_eta, P_gamma, 0.001, 1, .1, .1)

  if (is.null(lambdas)) {
    lambdas <- rep(1, 2 * L)
  } else if (length(lambdas) != 2 * L) {
    stop("lambdas must have length 2 * L.", call. = FALSE)
  }

  if (is.null(gammas)) {
    gammas <- rbinom(L, size = 1, prob = 0.5)
  } else if (length(gammas) != L) {
    stop("gammas must have length L.", call. = FALSE)
  }

  if (is.null(betas)) {
    betas <- rnorm(L) * gammas
  } else if (length(betas) != L) {
    stop("betas must have length L.", call. = FALSE)
  }

  if (is.null(Etas)) {
    Etas <- vector("list", L)
    for (l in seq_len(L)) {
      Eta_l <- matrix(0, nrow = p, ncol = p)

      if (gammas[l] == 1) {
        Eta_l <- matrix(sample(c(0, 1), p * p, replace = TRUE), nrow = p, ncol = p)
        Eta_l[!upper.tri(Eta_l)] <- 0
        Eta_l <- Eta_l + t(Eta_l)
        Eta_l <- Eta_l * As[[l]]
      }

      diag(Eta_l) <- sample(c(0, 1), p, replace = TRUE)
      Etas[[l]] <- Eta_l
    }
  } else if (length(Etas) != L) {
    stop("Etas must be a list of length L.", call. = FALSE)
  }

  inits <- list(constants, sigma2, lambdas, betas, gammas, Etas)

  start <- Sys.time()
  res <- run_mcmc(iterations, burn, L, As, Y, X, inits)
  end <- Sys.time()

  res$time <- end - start
  res
}
