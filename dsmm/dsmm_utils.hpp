namespace dsmm{

void pwise_dist2(double *A, double *B, int M, int N, int D, double *out);
void pwise_dist2_same(double *A, int M, int D, double *out);
void dot(double *A_arr, double *B_arr, int M, int N, int P, double *out_arr);
void dot_diag(double *A, double *B, int M, int P, double *out);
double digamma(double x, int order=5, int order2=10);
double logmenodigamma(double x, int order=5, int order2=10);
void studt(double *pwise_dist, int M, int N, double sigma2, double *Gamma, int D, double *out);
double fastexp(double x, int order=10);
void fastexp(double *X, int N, int order, double *out);
double fastlog(double x);
double eqforgamma(double x, double CDE_term);
void solveforgamma(double *X, int sizeX, double *out) ;
double eqforalpha(double alpha, double *p, int M, int N, double *sumPoverN);
void solveforalpha(double *p, int M, int N, double *sumPoverN, double &alpha);
void sumPoverN(double *pwise_dist, int M, int N, double neighbor_cutoff, double *p, double *sumPoverN);

}
