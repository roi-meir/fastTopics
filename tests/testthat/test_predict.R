context("predict")

test_that("project_poisson_nmf leaves the F matrix unchanged",{

  # Simulate a 160 x 100 counts matrix.
  set.seed(1)
  dat   <- simulate_multinom_gene_data(160,100,k = 3)
  train <- dat$X[1:80,]
  test  <- dat$X[81:160,]

  # Fit a Poisson non-negative matrix factorization using the
  # training data.
  capture.output(fit <- init_poisson_nmf(train,F = dat$F,
                                         init.method = "random"))
  capture.output(fit <- fit_poisson_nmf(train,fit0 = fit))
  F <- fit$F

  # Predict L in unseen (test) data points.
  capture.output(out <- project_poisson_nmf(test,F,numiter = 20))
  expect_equal(rep(1,3),unname(diag(cor(out$F,F))),tolerance = 1e-8)
})
