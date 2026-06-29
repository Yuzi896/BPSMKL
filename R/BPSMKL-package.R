#' BPSMKL package
#'
#' Bayesian pathway sparse multiple kernel learning.
#'
#' @useDynLib BPSMKL, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats rbinom rnorm
#' @importFrom utils combn
"_PACKAGE"

utils::globalVariables(c("Var1", "Var2", "value"))
