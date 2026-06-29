#' Summarize BPS-MKL MCMC results
#'
#' `res_summary()` summarizes posterior pathway and eta inclusion probabilities
#' from a fitted BPS-MKL result object. It always returns an `idx_res` data
#' frame with pathway labels, predictor index pairs, and estimated eta
#' probabilities. When simulated truth is supplied through `pathway`, it also
#' returns pathway-level AUC values. When `Figure = TRUE` and `ggplot2` is
#' available, it also returns diagnostic heatmaps.
#'
#' @param res Result list returned by `bps_mkl()` or `bps_mkl_app()`.
#' @param pathway Optional simulated pathway object, such as the output from
#'   `simulate_pathway()`. When supplied, `pathway$A_imp` is used as truth for
#'   AUC calculation and optional figures.
#' @param gammas Optional numeric vector of true pathway inclusion indicators.
#'   When supplied with `pathway` and `Figure = TRUE`, truth heatmaps multiply
#'   `A_imp` by `gammas`.
#' @param p Optional integer number of predictors. If `NULL`, this is inferred
#'   from `dim(res$eta_whole[[1]])[1]`.
#' @param burnin Integer number of initial MCMC draws to discard. Defaults to
#'   `0`.
#' @param Figure Logical. If `TRUE`, return heatmap figures when `ggplot2` is
#'   available. Defaults to `FALSE`.
#'
#' @return A list containing:
#' \describe{
#'   \item{`idx_res`}{A data frame with `pathway_id`, `idx1`, `idx2`,
#'   `estimated`, and `P_pathway`. If `pathway` is supplied, it also contains
#'   `truth`.}
#'   \item{`auc_sim`}{A data frame with pathway-level AUC values. Returned only
#'   when `pathway` is supplied.}
#'   \item{`Figure`}{A list of pathway heatmaps. Returned only when
#'   `Figure = TRUE` and `ggplot2` is available. If `pathway` is supplied,
#'   truth heatmaps are included.}
#' }
#'
#' @export
res_summary <- function(res,
                        pathway = NULL,
                        gammas = NULL,
                        p = NULL,
                        burnin = 0, Figure = FALSE) {
  L <- length(res$eta_whole)
  N <- nrow(res$gamma_save)
  if (burnin < 0 || burnin >= N) {
    stop("burnin must be between 0 and nrow(res$gamma_save) - 1.", call. = FALSE)
  }
  keep <- seq.int(burnin + 1, N)

  if (is.null(p)) {
    p <- dim(res$eta_whole[[1]])[1]
  }

  has_truth <- !is.null(pathway) && !is.null(gammas)
  gamma_prob <- numeric(L)
  idx_res_list <- vector("list", L)
  eta_prob <- vector("list", L)
  auc_values <- rep(NA, L)

  for (i in seq_len(L)) {
    gamma_draws <- res$gamma_save[keep, i]
    eta_draws <- res$eta_whole[[i]][keep, , drop = FALSE]
    active_eta <- rowSums(eta_draws) > 1
    gamma_prob[i] <- mean(gamma_draws * active_eta)

    eta_given_pathway <- eta_draws[gamma_draws == 1, , drop = FALSE]
    if (nrow(eta_given_pathway) == 0) {
      eta_est <- rep(0, ncol(eta_draws))
    } else {
      eta_est <- colMeans(eta_given_pathway)
    }

    idx <- as.data.frame(res$eta_idx[[i]])
    names(idx) <- c("idx1", "idx2")

    eta_mat_vec <- rep(0, p * p)
    for (kk in seq_len(nrow(idx))) {
      idx1 <- idx$idx1[kk]
      idx2 <- idx$idx2[kk]
      row_idx <- (idx1 - 1) * p + idx2
      row_idx_t <- idx1 + (idx2 - 1) * p
      eta_mat_vec[row_idx] <- eta_est[kk]
      eta_mat_vec[row_idx_t] <- eta_est[kk]
    }
    eta_prob[[i]] <- eta_mat_vec

    idx_res_i <- data.frame(
      pathway_id = i,
      idx1 = idx$idx1,
      idx2 = idx$idx2,
      estimated = eta_est,
      P_pathway = gamma_prob[i]
    )
    if (has_truth) {
      truth <- rep(0, nrow(idx))
      for (kk in seq_len(nrow(idx))) {
        truth[kk] <- pathway$A_imp[[i]][idx$idx1[kk], idx$idx2[kk]]
      }
      idx_res_i$truth <- truth

      if (length(unique(truth)) >= 2) {
        if (requireNamespace("pROC", quietly = TRUE)) {
          rocc <- pROC::roc(truth, eta_est, direction = "<", quiet = TRUE)
          auc_values[i] <- as.numeric(pROC::auc(rocc))
        } else {
          pos <- eta_est[truth == 1]
          neg <- eta_est[truth == 0]
          auc_values[i] <- NA
        }
      }
    }

    idx_res_list[[i]] <- idx_res_i
  }

  out <- list(idx_res = do.call(rbind, idx_res_list))

  if (has_truth) {
    out$auc_sim <- data.frame(pathway = seq_len(L), auc = auc_values)
  }

  if (isTRUE(Figure) &&
      requireNamespace("ggplot2", quietly = TRUE)) {

    true_gamma <- if (is.null(gammas)) rep(1, L) else gammas
    figs <- vector("list", L)

    for (k in seq_len(L)) {
      idx_sub <- sort(unique(c(res$eta_idx[[k]])))
      mat_eta <- matrix(eta_prob[[k]], nrow = p, ncol = p)

      eta_df <- data.frame(
        expand.grid(Var1 = idx_sub, Var2 = idx_sub),
        value = as.vector(mat_eta[idx_sub, idx_sub])
      )

      p1 <- ggplot2::ggplot(
        eta_df,
        ggplot2::aes(factor(Var1), factor(Var2), fill = value)
      ) +
        ggplot2::geom_tile() +
        ggplot2::scale_fill_viridis_c(limits = c(0, 1)) +
        ggplot2::coord_fixed() +
        ggplot2::labs(
          title = paste0("Inferred Eta[[", k, "]] ","Pathway Prob: ",gamma_prob[k]),
          x = "Node i",
          y = "Node j"
        ) +
        ggplot2::theme_minimal()

      if (has_truth) {
        truth_df <- data.frame(
          expand.grid(Var1 = idx_sub, Var2 = idx_sub),
          value = as.vector(pathway$A_imp[[k]][idx_sub, idx_sub]) * true_gamma[k]
        )

        p2 <- ggplot2::ggplot(
          truth_df,
          ggplot2::aes(factor(Var1), factor(Var2), fill = factor(value))
        ) +
          ggplot2::geom_tile() +
          ggplot2::scale_fill_manual(values = c("white", "black")) +
          ggplot2::coord_fixed() +
          ggplot2::labs(
            title = paste0("Truth for Pathway ", k),
            x = "Node i",
            y = "Node j"
          ) +
          ggplot2::theme_minimal() +
          ggplot2::theme(legend.position = "none")

        if (requireNamespace("patchwork", quietly = TRUE)) {
          figs[[k]] <- p1 + p2 + patchwork::plot_layout(guides = "collect")
        } else {
          figs[[k]] <- list(estimated = p1, truth = p2)
        }
      } else {
        figs[[k]] <- p1
      }
    }

    out$Figure <- figs
  }

  out
}
