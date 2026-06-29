#' Run approximate BPS-MKL from individual initial values
#'
#' `bps_mkl_app()` collects the individual initial values into the list expected
#' by the optimized C++ sampler and then runs the approximate Nyström BPS-MKL
#' algorithm.
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
#' @param m Integer. Number of observations, or rows of `X`, to sample as
#'   Nyström landmarks.
#' @param P_eta Numeric scalar. eta inclusion probability.
#' @param P_gamma Numeric scalar. gamma inclusion probability.
#' @param sigma2 Numeric scalar. initial residual variance.
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
#' # Assume simulation objects are already pre-generated.
#' # See details in the `bps_mkl()` example.
#' result <- bps_mkl_app(
#'   iterations = N,
#'   burn = burnin,
#'   L = L,
#'   As = A,
#'   Y = Y,
#'   X = X,
#'   m = 50,
#'   sigma2 = sigma2,
#'   betas = beta,
#'   gammas = gamma,
#'   Etas = A_imp
#' )
#' }
#'
#' @export
bps_mkl_app <- function(iterations, burn, L, As, Y, X, m,
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

  n <- nrow(X)
  p <- ncol(X)
  if (length(As) != L) {
    stop("As must be a list of length L.", call. = FALSE)
  }
  if (length(m) != 1L || is.na(m) || m < 1L || m > n) {
    stop("m must be a single integer between 1 and nrow(X).", call. = FALSE)
  }

  m <- as.integer(m)
  landmark_idx <- sort(sample(seq_len(n), m))

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

  # check if diagonal element for each pathway has 1s
  for(l in 1:L){
    # if no nodes are one
    if(sum(diag(As[[l]]))==0){
      nodes = unique(c(which(As[[l]]==1,arr.ind = TRUE)))
      # force all the nodes appear at least once on edges being considered for importance analysis
      As[[l]][cbind(nodes, nodes)] <- 1
    }
  }

  inits <- list(constants, sigma2, lambdas, betas, gammas, Etas)

  start <- Sys.time()
  res <- run_mcmc_opt(iterations, burn, L, As, Y, X, inits, landmark_idx)
  end <- Sys.time()

  res$time <- end - start
  res
}
