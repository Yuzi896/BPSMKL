#' Simulate pathway adjacency matrices
#'
#' `simulate_pathway()` creates non-overlapping simulated pathways over `p`
#' features. For each pathway, it samples `pathway_p` features, creates
#' `n_edges` candidate interaction edges among those features, and marks
#' `n_imp_edges` of those candidate edges as important. Optionally, it can also
#' add important main effects on the diagonal of each important-effect matrix.
#'
#' @param p Integer. Total number of features.
#' @param n_pathways Integer. Number of pathways to simulate.
#' @param pathway_p Integer. Number of features assigned to each pathway.
#' @param n_edges Integer. Number of candidate interaction edges to include in
#'   each pathway adjacency matrix.
#' @param n_imp_edges Integer. Number of candidate interaction edges to mark as
#'   important in each pathway.
#' @param n_imp_main Integer. Number of pathway features to mark as important
#'   main effects when `main = TRUE`. Defaults to `0`.
#' @param main Logical. If `TRUE`, add important main effects on the diagonal of
#'   each important-effect matrix. Defaults to `FALSE`.
#' @param seed Optional integer random seed used before simulation. Defaults to
#'   `NULL`.
#'
#' @return A list with two elements:
#' \describe{
#'   \item{`A`}{A list of `n_pathways` `p x p` adjacency matrices containing
#'   the candidate interaction edges for each pathway.}
#'   \item{`A_imp`}{A list of `n_pathways` `p x p` matrices containing the
#'   important interaction edges and, when requested, important main effects.}
#' }
#'
#' @export
simulate_pathway <- function(p,
                             n_pathways,
                             pathway_p,
                             n_edges,
                             n_imp_edges,
                             n_imp_main=0,
                             main=FALSE,
                             seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  # all possible unique features
  available_features <- c(1:p)

  # sanity check: make sure we’re not asking for more unique edges than possible
  if (n_pathways*pathway_p > p) {
    stop("Not enough unique features to create non-overlapping pathways.")
  }

  pathways <- list(n_pathways)

  # initialize storage
  A_list <- list(n_pathways)
  A_imp_list <-  list(n_pathways)
  W_list <- list(n_pathways)
  start_idx <- 1

  for (g in seq_len(n_pathways)) {
    # allocate disjoint edges for this pathway
    # path_feature <- sort(sample(seq_along(available_features), pathway_p))
    path_feature <- sort(sample(available_features, pathway_p))
    available_features <- setdiff(available_features, path_feature)


    path_edge <- combn(length(path_feature), 2, simplify = FALSE)
    path_edge_idx <- sample(seq_along(path_edge),n_edges)
    imp_idx <- sample(seq_along(path_edge_idx),n_imp_edges)

    # # now we recorde path_features
    # available_features <- setdiff(available_features,path_feature)

    edges <- path_edge[path_edge_idx]
    imp_edges <- edges[imp_idx]

    # build adjacency matrices
    A <- matrix(0, p, p)
    for (e in edges) A[path_feature[e[1]], path_feature[e[2]]] <- 1
    A[!upper.tri(A)] <- 0
    A <- A + t(A)

    # add selfloop
    # diag(A) = 1

    A_imp <- matrix(0, p, p)
    for (e in imp_edges) A_imp[path_feature[e[1]], path_feature[e[2]]] <- 1
    A_imp[!upper.tri(A_imp)] <- 0
    A_imp <- A_imp + t(A_imp)

    # add some main effect
    if(main == TRUE){
      A_imp_main = sample(path_feature,n_imp_main)
      if(length(A_imp_main) > 0){
        for(e in A_imp_main) A_imp[e, e] <- 1
      }
    }



    # weights
    # w <- A_imp + A * 0.2
    # w  <- A

    A_list[[g]] <- A
    A_imp_list[[g]] <- A_imp
    # W_list[[g]] <- w

  }

  list(
    A = A_list,
    A_imp = A_imp_list
  )
}
