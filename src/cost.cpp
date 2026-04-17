#include <RcppParallel.h>
#include "misc.h"
#include "cost.h"

using namespace arma;

// CLASS DEFINITIONS
// -----------------
// Worker for parallel computation of the dense cost function using
// RcppParallel reduce. Each thread accumulates its own partial sum
// over a range of columns of X, and the join() method combines them.
struct cost_worker : public RcppParallel::Worker {
  const mat&  X;
  const mat&  A;
  const mat&  B;
  double      e;
  bool        poisson;
  vec         f;

  cost_worker (const mat& X, const mat& A, const mat& B, double e,
               bool poisson) :
    X(X), A(A), B(B), e(e), poisson(poisson), f(X.n_rows, fill::zeros) { };

  // Split constructor: creates a new worker with a zeroed accumulator.
  cost_worker (const cost_worker& other, RcppParallel::Split) :
    X(other.X), A(other.A), B(other.B), e(other.e), poisson(other.poisson),
    f(other.X.n_rows, fill::zeros) { };

  // Accumulate contributions from columns [begin, end).
  void operator() (std::size_t begin, std::size_t end) {
    vec y(A.n_rows);
    for (unsigned int j = begin; j < end; j++) {
      y  = A * B.col(j);
      f -= X.col(j) % log(y + e);
      if (poisson)
        f += y;
    }
  }

  void join (const cost_worker& other) { f += other.f; }
};

// Worker for parallel computation of the sparse cost function.
struct cost_sparse_worker : public RcppParallel::Worker {
  const sp_mat& X;
  const mat&    A;
  const mat&    B;
  double        e;
  bool          poisson;
  vec           f;

  cost_sparse_worker (const sp_mat& X, const mat& A, const mat& B, double e,
                      bool poisson) :
    X(X), A(A), B(B), e(e), poisson(poisson), f(X.n_rows, fill::zeros) { };

  cost_sparse_worker (const cost_sparse_worker& other, RcppParallel::Split) :
    X(other.X), A(other.A), B(other.B), e(other.e), poisson(other.poisson),
    f(other.X.n_rows, fill::zeros) { };

  void operator() (std::size_t begin, std::size_t end) {
    unsigned int i;
    vec y(A.n_rows);
    for (unsigned int j = begin; j < end; j++) {
      sp_mat::const_col_iterator xj = X.begin_col(j);
      sp_mat::const_col_iterator xm = X.end_col(j);
      y = A * B.col(j);
      for (; xj != xm; ++xj) {
        i     = xj.row();
        f(i) -= (*xj) * log(y(i) + e);
      }
      if (poisson)
        f += y;
    }
  }

  void join (const cost_sparse_worker& other) { f += other.f; }
};

// FUNCTION DEFINITIONS
// --------------------
// Compute negative log-likelihoods for assessing a topic model fit or
// quality of a non-negative matrix factorization, in which matrix X
// is approximated by matrix product A * B.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec cost_rcpp (const arma::mat& X, const arma::mat& A,
		     const arma::mat& B, double e, bool poisson) {
  return cost(X,A,B,e,poisson);
}

// This is the same as cost_rcpp, except that X must be sparse.
//
// [[Rcpp::export]]
arma::vec cost_sparse_rcpp (const arma::sp_mat& X, const arma::mat& A,
	  		    const arma::mat& B, double e, bool poisson) {
  return cost_sparse(X,A,B,e,poisson);
}

// This is the helper function for cost_rcpp.
arma::vec cost (const mat& X, const mat& A, const mat& B, double e, 
             bool poisson) {
  unsigned int n = X.n_rows;
  unsigned int m = X.n_cols;
  vec  f(n,fill::zeros);
  vec  y(n);
  
  // Repeat for each column of X.
  for (unsigned int j = 0; j < m; j++) {

    // This is equivalent to the following R code:
    //
    //   f = f + poisson*y - X[,j]*log(y + e))
    //
    // where 
    // 
    //   y = A %*% B[,j]
    //
    y  = A * B.col(j);
    f -= X.col(j) % log(y + e);
    if (poisson)
      f += y;
  }
  
  return f;
}

// This is the same as cost_rcpp, except that Intel Threading Building
// Blocks (TBB) are used to compute the contributions from each column
// of X in parallel.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::export]]
arma::vec cost_parallel_rcpp (const arma::mat& X, const arma::mat& A,
			      const arma::mat& B, double e, bool poisson) {
  cost_worker worker(X,A,B,e,poisson);
  parallelReduce(0,X.n_cols,worker);
  return worker.f;
}

// This is the same as cost_sparse_rcpp, except that Intel Threading
// Building Blocks (TBB) are used to compute the contributions from
// each column of X in parallel.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::export]]
arma::vec cost_sparse_parallel_rcpp (const arma::sp_mat& X, const arma::mat& A,
				     const arma::mat& B, double e, bool poisson) {
  cost_sparse_worker worker(X,A,B,e,poisson);
  parallelReduce(0,X.n_cols,worker);
  return worker.f;
}

// Helper function for cost_sparse_rcpp.
arma::vec cost_sparse (const sp_mat& X, const mat& A, const mat& B,
		       double e, bool poisson) {
  unsigned int n = X.n_rows;
  unsigned int m = X.n_cols;
  unsigned int i;
  vec  f(n,fill::zeros);
  vec  y(n);
  
  // Repeat for each column of X.
  for (unsigned int j = 0; j < m; j++) {

    // Initialize an iterator for the nonzero elements in the jth
    // column of X.
    sp_mat::const_col_iterator xj = X.begin_col(j);
    sp_mat::const_col_iterator xm = X.end_col(j);

    // This is equivalent to the following R code:
    //
    //   f = f + poisson*y - X[,j]*log(y + e)
    //
    // where 
    // 
    //   y = A %*% B[,j]
    //
    y = A * B.col(j);
    for(; xj != xm; ++xj) {
      i     = xj.row();
      f(i) -= (*xj) * log(y(i) + e);
    }
    if (poisson)
      f += y;
  }
  
  return f;
}
