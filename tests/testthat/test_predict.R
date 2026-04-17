context("predict")

# Test that project_poisson_nmf returns F with the same structure as the input F
test_that("project_poisson_nmf returns F with the same structure as the input F", {

  set.seed(1)
  dat   <- simulate_multinom_gene_data(175,1200,k = 3)
  train <- dat$X[1:100,]
  test  <- dat$X[101:175,]

  fit    <- init_poisson_nmf(train,F = dat$F,init.method = "random")
  fit    <- fit_poisson_nmf(train,fit0 = fit,verbose = "none")
  fit    <- poisson2multinom(fit)
  F_pois <- multinom2poisson(fit)$F

  out <- project_poisson_nmf(test,F_pois,numiter = 20,verbose = "none", update.factors = NULL)

  norm_col <- function(M) sweep(M,2,colSums(M),"/")
  expect_equal(norm_col(out$F),norm_col(F_pois),tolerance = 1e-10)
})
