#include <RcppParallel.h>
#include "misc.h"

using namespace arma;

// CLASS DEFINITIONS
// -----------------
// Worker for computing t(A) %*% L and A %*% F simultaneously in one
// pass over the nonzeros of X, where A[i,j] = X[i,j] / (dot(L[i,:],
// F[j,:]) + e). L is n x k, F is m x k.
//
// t(A) %*% L is m x k: row j = sum_{i: X[i,j]!=0} A[i,j] * L[i,:]
//   Each column j of X writes only to row j -> parallelFor-safe.
//
// A %*% F is n x k: row i = sum_{j: X[i,j]!=0} A[i,j] * F[j,:]
//   Multiple columns can write to the same row -> needs reduction.
//
// Both are accumulated in a single parallelReduce over columns of X.
struct kkt_sparse_worker : public RcppParallel::Worker {
  const sp_mat& X;
  const mat&    L;
  const mat&    F;
  double        e;
  mat           tAL;  // thread-local m x k accumulator for t(A) %*% L
  mat           AF;   // thread-local n x k accumulator for A %*% F

  kkt_sparse_worker (const sp_mat& X, const mat& L, const mat& F, double e) :
    X(X), L(L), F(F), e(e),
    tAL(X.n_cols, L.n_cols, fill::zeros),
    AF(X.n_rows,  F.n_cols, fill::zeros) { };

  kkt_sparse_worker (const kkt_sparse_worker& other, RcppParallel::Split) :
    X(other.X), L(other.L), F(other.F), e(other.e),
    tAL(other.X.n_cols, other.L.n_cols, fill::zeros),
    AF(other.X.n_rows,  other.F.n_cols, fill::zeros) { };

  void operator() (std::size_t begin, std::size_t end) {
    for (unsigned int j = begin; j < end; j++) {
      sp_mat::const_col_iterator xi = X.begin_col(j);
      sp_mat::const_col_iterator xm = X.end_col(j);
      rowvec Fj = F.row(j);
      for (; xi != xm; ++xi) {
        unsigned int i = xi.row();
        double       a = (*xi) / (dot(L.row(i), Fj) + e);
        tAL.row(j) += a * L.row(i);
        AF.row(i)  += a * Fj;
      }
    }
  }

  void join (const kkt_sparse_worker& other) {
    tAL += other.tAL;
    AF  += other.AF;
  }
};

// Worker for parallel computation of x_over_crossprod. Each element
// t of the output is independent: y(t) = x(t) / (dot(A.col(i(t)),
// B.col(j(t))) + e). parallelFor is safe here because each thread
// writes to a disjoint range of y.
struct x_over_crossprod_worker : public RcppParallel::Worker {
  const vec&  i_idx;
  const vec&  j_idx;
  const mat&  A;
  const mat&  B;
  double      e;
  vec&        y;

  x_over_crossprod_worker (const vec& i_idx, const vec& j_idx, const mat& A,
                            const mat& B, double e, vec& y) :
    i_idx(i_idx), j_idx(j_idx), A(A), B(B), e(e), y(y) { };

  void operator() (std::size_t begin, std::size_t end) {
    for (unsigned int t = begin; t < end; t++)
      y(t) /= (dot(A.col((unsigned int) i_idx(t)),
                   B.col((unsigned int) j_idx(t))) + e);
  }
};

// FUNCTION DECLARATIONS
// ---------------------
void le_diff (const vec& x, vec& y);

// FUNCTION DEFINITIONS
// --------------------
// Compute, for each row of X, the "least extreme" differences. This
// should output the same result as t(apply(X,1,le.diff)), but faster.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat le_diff_rcpp (const arma::mat& X) {
  unsigned int n = X.n_rows;
  unsigned int m = X.n_cols;
  mat Y(n,m);
  vec x(m);
  vec y(m);
  for (unsigned int i = 0; i < n; i++) {
    x = trans(X.row(i));
    le_diff(x,y);
    Y.row(i) = trans(y);
  }
  return Y;
}

// This is used to implement x_over_tcrossprod.
//
// [[Rcpp::export]]
arma::vec x_over_crossprod_rcpp (const arma::vec& i, const arma::vec& j,
				 const arma::vec& x, const arma::mat& A,
				 const arma::mat& B, double e) {
  unsigned int n = x.n_elem;
  vec y = x;
  for (unsigned int t = 0; t < n; t++)
    y(t) /= (dot(A.col(i(t)),B.col(j(t))) + e);
  return y;
}

// This is the same as x_over_crossprod_rcpp, except that Intel
// Threading Building Blocks (TBB) are used to process the nonzero
// elements in parallel.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::export]]
arma::vec x_over_crossprod_parallel_rcpp (const arma::vec& i,
					  const arma::vec& j,
					  const arma::vec& x,
					  const arma::mat& A,
					  const arma::mat& B, double e) {
  vec y = x;
  x_over_crossprod_worker worker(i,j,A,B,e,y);
  parallelFor(0,x.n_elem,worker);
  return y;
}

// Compute KKT residuals for the sparse Poisson NMF problem in
// parallel. Returns a list with two matrices:
//   tAL: t(A) %*% L (m x k), where A[i,j] = X[i,j]/(dot(L[i,:],F[j,:])+e)
//   AF:  A %*% F    (n x k)
// This fuses the x_over_tcrossprod, t(A)%*%L, and A%*%F computations
// into a single parallel pass over the nonzeros of X.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::export]]
Rcpp::List poisson_nmf_kkt_sparse_parallel_rcpp (const arma::sp_mat& X,
						  const arma::mat& L,
						  const arma::mat& F,
						  double e) {
  kkt_sparse_worker worker(X,L,F,e);
  parallelReduce(0,X.n_cols,worker);
  return Rcpp::List::create(Rcpp::Named("tAL") = worker.tAL,
			    Rcpp::Named("AF")  = worker.AF);
}

// For vector x, return a vector of the same length y containing the
// "least extreme" differences y(i) = x(i) - x(j), in which j is the
// index not equal to i such that abs(x(i) - x(j)) is the smallest
// possible.
void le_diff (const vec& x, vec& y) {
  unsigned int n = x.n_elem;
  if (n == 2) {
    y(0) = x(0) - x(1);
    y(1) = -y(0);
  } else {
    uvec indices = sort_index(x);
    unsigned int i, j, k;
    double a, b;
    i = indices(0);
    j = indices(1);
    y(i) = x(i) - x(j);
    i = indices(n-1);
    j = indices(n-2);
    y(i) = x(i) - x(j);
    for (unsigned int t = 1; t < n-1; t++) {
      i = indices(t-1);
      j = indices(t);
      k = indices(t+1);
      a = x(j) - x(i);
      b = x(k) - x(j);
      if (a <= b)
        y(j) = x(j) - x(i);
      else
        y(j) = x(j) - x(k);
    }
  }
}

// Return the row indices of the nonzeros in the jth column of sparse
// matrix A. This is the same as
//
//  i = find(A.col(j))
//
// but this code does not compile in some versions of gcc, so I
// re-implemented this code here. Vector i must already been
// initialized with the proper length, e.g., by doing
//
//   vec a = nonzeros(A.col(j));
//   unsigned int n = a.n_elem;
//   uvec i(n);
//   getcolnonzeros(A,i,j);
//
void getcolnonzeros (const sp_mat& A, uvec& i, unsigned int j) {
  sp_mat::const_col_iterator ai = A.begin_col(j);
  sp_mat::const_col_iterator an = A.end_col(j);
  for (unsigned int t = 0; ai != an; ++ai, ++t)
    i(t) = ai.row();
}

// Scale each column A[,j] by b[j].
void scalecols (mat& A, const vec& b) {
  unsigned int n = A.n_rows;
  unsigned int m = A.n_cols;
  for (unsigned int j = 0; j < m; j++)
    for (unsigned int i = 0; i < n; i++)
      A(i,j) *= b(j);
}

// Normalize each row of A so that the entries in each row sum to 1.
void normalizerows (mat& A) {
  unsigned int n = A.n_rows;
  unsigned int m = A.n_cols;
  vec b = conv_to<vec>::from(sum(A,1));
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int j = 0; j < m; j++)
      A(i,j) /= b(i);
}

// Normalize each column of A so that the entries in each column sum to 1.
void normalizecols (mat& A) {
  unsigned int n = A.n_rows;
  unsigned int m = A.n_cols;
  rowvec b = sum(A,0);
  for (unsigned int j = 0; j < m; j++)
    for (unsigned int i = 0; i < n; i++)
      A(i,j) /= b(j);
}

// Scale each row of A so that the largest entry in each row is 1.
void normalizerowsbymax (mat& A) {
  unsigned int n = A.n_rows;
  unsigned int m = A.n_cols;
  vec b = conv_to<vec>::from(max(A,1));
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int j = 0; j < m; j++)
      A(i,j) /= b(i);
}
