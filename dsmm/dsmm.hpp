namespace dsmm{

void _dsmm(double *X, double *Y, int M, int N, int D,
           double beta, double lambda, double neighbor_cutoff,
           double alpha, double conv_epsilon,
           double *pwise_dist, double *pwise_distYY,
           double *Gamma, double *CDE_term,
           double *w, double *F_t, double *wF_t, double *wF_t_sum, 
           double *p, double *u,
           double *hatP, double *hatPI_diag, double *hatPIG, double *hatPX, double *hatPIY,
           double *G, double *W, double *GW, 
           double *sumPoverN, double *expAlphaSumPoverN);
           
}
