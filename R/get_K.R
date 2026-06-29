#' Construct a dense kernel matrix for selected effects
#'
#' `get_K()` builds the `N` by `N` kernel matrix used by the model from an
#' input design matrix, an effect-selection matrix, and two bandwidth
#' parameters. The candidate main-effect and interaction index matrix is
#' computed internally from `Eta`.
#'
#' Off-diagonal active entries in `Eta` are treated as interactions and use
#' `lambda[1]`; diagonal active entries are treated as main effects and use
#' `lambda[2]`. When `update = TRUE`, the C++ helper works with a temporary copy
#' of `Eta` where `Eta[i, j]` and `Eta[j, i]` are set to `eta`.
#' The candidate index matrix is always computed from the original `Eta`.
#'
#' @param X Numeric matrix with observations in rows and predictors in columns.
#' @param Eta Numeric square matrix indicating which main effects and
#'   interactions are active. Diagonal entries represent main effects and
#'   off-diagonal entries represent interactions.
#' @param lambda Numeric vector of length 2. `lambda[1]` is the bandwidth for
#'   interaction terms and `lambda[2]` is the bandwidth for main effects.
#' @param update Logical flag. Use `FALSE` to compute the kernel from `Eta` as
#'   supplied, or `TRUE` to compute it after temporarily updating the `(i, j)`
#'   and `(j, i)` entries of `Eta`.
#' @param eta Proposed value to use when `update = TRUE`; must be `0` or `1`.
#'   Defaults to `NULL`.
#' @param i One-based row index in `Eta` to update when `update = TRUE`.
#'   Defaults to `NULL`.
#' @param j One-based column index in `Eta` to update when `update = TRUE`.
#'   Defaults to `NULL`.
#'
#' @return An `N` by `N` numeric kernel matrix, where `N = nrow(X)`.
#'
#' @export
get_K <- function(X, Eta, lambda, update = FALSE, eta = NULL, i = NULL, j = NULL) {
  update_int <- as.integer(isTRUE(update))
  if (update_int == 0L) {
    eta <- -1L
    i <- -1L
    j <- -1L
  } else {
    if (is.null(eta) || is.null(i) || is.null(j)) {
      stop("eta, i, and j must be supplied when update = TRUE.", call. = FALSE)
    }
    if (!eta %in% c(0, 1)) {
      stop("eta must be 0 or 1.", call. = FALSE)
    }

    eta <- as.integer(eta)
    i <- as.integer(i) - 1L
    j <- as.integer(j) - 1L
  }

  idx <- extract_edges_fast(Eta)
  get_K_cpp(update_int, X, Eta, lambda, idx, eta, i, j)
}
