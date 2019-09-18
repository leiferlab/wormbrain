#include "dsmm_utils.hpp"
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/digamma.hpp>

using namespace dsmm;

class tolerance {
public:
	tolerance(double eps) :
		_eps(eps) {
	}
	bool operator()(double a, double b) {
		return (fabs(b - a) <= _eps);
	}
private:
	double _eps;
};

void dsmm::pwise_dist2(double *A, double *B, int M, int N, int D, double *out) {
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            out[m*N+n] = 0.0;
            for(int d=0;d<D;d++){
                out[m*N+n] += pow(A[m*D+d]-B[n*D+d],2);
            }
        }
    }
}

void dsmm::pwise_dist2_same(double *A, int M, int D, double *out) {
    double dist;
    for(int i=0;i<M;i++){
        for(int j=i;j<M;j++){
            out[i*M+j] = 0.0;
            out[j*M+i] = 0.0;
            for(int d=0;d<D;d++){
                dist = pow(A[i*D+d]-A[j*D+d],2);
                out[i*M+j] += dist;
                out[j*M+i] += dist;
            }
        }
    }
}

void dsmm::dot(double *A_arr, double *B_arr, int M, int N, int P, double *out_arr) {   
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
    typedef Eigen::Map<MatrixType> MapType;
    
    MapType A(A_arr,M,N), B(B_arr,N,P), out(out_arr,M,P);
    
    out.noalias() = A*B;    
}

void dsmm::dot_diag(double *A, double *B, int M, int P, double *out) {   
    // A MxM diagonal, represented as the vector of the diagonal elements
    // B MxP
        
    double am = 0.0;
    
    for(int m=0;m<M;m++){
        am = A[m];
        for(int p=0;p<P;p++){
            out[m*P+p] = B[m*P+p]*am;
        }
    }
        
}

void dsmm::fastexp(double *X, int N, int order, double *out) { //FIXME this implementation is slower than the regular exp! See implementation below
    double a;
    a = pow(2.0,order);
    for(int n=0;n<N;n++){
        out[n] = pow(1.+X[n]/a,n);
    }
}

double dsmm::fastexp(double x, int order) {
    //return pow(1.+x/pow(2.0,order),order);
    double y;
    y = 1.0 + x / 4096.0;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    return y;
}

double dsmm::fastlog(double x) {
    //return pow(1.+x/pow(2.0,order),order);
    double y;
    y = (x-1.)/(x+1.);
    y = 2.0*(y+0.3333333*y*y*y+0.2*y*y*y*y*y);
    return y;
}  


double dsmm::digamma(double x, int order, int order2) {
    double coeff[15] = {-0.5,-1./12.,0.0,1./120.,0.0,-1./256.,0.0,1./240.,0.0,-5./660.,0.0,691./32760.,0.0,-1./12.};
    
    double xporder2 = x+order2;
    double y = log(xporder2); 
    
    for(int j=0;j<order2;j++) {
        y -= 1./(x+j);
    }
    
    for (int k=0;k<order;k++) {
        if(coeff[k]!=0.0){
            y += coeff[k]/xporder2;
            //y += coeff[k]/pow(xporder2,(double)k+1.); // VERSION 1
        }
        xporder2 *= xporder2;
    }
    
    return y;
}

double dsmm::logmenodigamma(double x, int order, int order2) {
    double coeff[15] = {-0.5,-1./12.,0.0,1./120.,0.0,-1./256.,0.0,1./240.,0.0,-5./660.,0.0,691./32760.,0.0,-1./12.};
    
    double xporder2 = x+order2;
    double y = log(x/xporder2); 
    
    for(int j=0;j<order2;j++) {
        y += 1./(x+j);
    }
    
    for (int k=0;k<order;k++) {
        if(coeff[k]!=0.0){
            y -= coeff[k]/xporder2;
            //y += coeff[k]/pow(xporder2,(double)k+1.); // VERSION 1
        }
        xporder2 *= xporder2;
    }
    
    return y;
}


void dsmm::studt(double *pwise_dist, int M, int N, double sigma2, double *Gamma, int D, double *out){
    // With Mahalanobis distance = pwise_dist/sigma^2
    double a,b,c,d,e,f;
    double sigma;
    double g05 = 1.772453850906; //gamma(0.5)=sqrt(pi)
    
    sigma = sqrt(sigma2);
    for(int m=0;m<M;m++){
        a = tgamma(0.5*(Gamma[m]+D));
        b = sigma * pow(Gamma[m]+g05,0.5*D) * tgamma(0.5*Gamma[m]);
        d = 1.0/(sigma2*Gamma[m]);
        e = 0.5*(D+Gamma[m]);
        f = a/b;
        for(int n=0;n<N;n++){
            c = pow(1.0 + pwise_dist[m*N+n]*d,e);
            out[m*N+n] = f/c;
        }
    }
}


double dsmm::eqforgamma(double x, double CDE_term) {
    double z = 0.5*x;
    //double A = -boost::math::digamma(z) + log(z);
    //double A = -digamma(z,5,10) + log(z);
    double A = logmenodigamma(z,5,10);
    return A+CDE_term+1.;
}

double dsmm::eqforgamma_mod(double x, double CDE_term) {
    double z = 0.5*abs(x);
    double A = -boost::math::digamma(z) + log(z);
    //double A = -digamma(z,5,10) + log(z);
    //double A = logmenodigamma(z,5,10);
    return A+CDE_term+1.;
}

void dsmm::solveforgamma(double *X, int sizeX, double *out, double eq_tol) {
    double xi;
    double inizio = 10.0;
    double fattore = 2.0;
    bool sale = false; //the function decreases with increasing Gamma
    tolerance tolleranza = eq_tol;
    boost::uintmax_t massimo = 1000;
    
    for(int i=0;i<sizeX;i++) {
        xi = X[i];
        //inizio = X[i];
        std::pair<double, double> result = boost::math::tools::bracket_and_solve_root(std::bind(eqforgamma,std::placeholders::_1,xi), inizio, fattore, sale, tolleranza, massimo); //FIXME TODO
        //std::pair<double, double> result = boost::math::tools::toms748_solve(std::bind(eqforgamma,std::placeholders::_1,xi),0.0,20.,tolleranza,massimo);
        out[i] = 0.5*(result.first + result.second);
    }
}

double dsmm::eqforalpha(double alpha, double *p, int M, int N, double *sumPoverN){
    //Eq. (20)', switching indices for x and y to match their first paper. 
    //Added an alpha to the first term inside the \biggl ( in the final line!!
    //Is it reasonable to use an approximate form for the exponential?
    //print(alpha*np.max(sumPoverN)) #around 0.2 max
    //print(alpha*np.min(sumPoverN)) # 0 (always positive, right?)
    double y = 0.0;
    
    double a;
    double b;
    double c;
    double d = 0.0;
    for(int m=0;m<M;m++){
        b = 0.0;
        c = 0.0;
        for(int n=0;n<N;n++){
            a = dsmm::fastexp(alpha*sumPoverN[m*N+n],3);
            //a = exp(alpha*sumPoverN[m*N+n]);
            b += sumPoverN[m*N+n] * a;
            c += a;
        }
        d = -b/c;
        for(int n=0;n<N;n++){
            y += p[m*N+n]*(alpha*sumPoverN[m*N+n]+d);
        }
    }
    
    return y;
}

void dsmm::solveforalpha(double *p, int M, int N, double *sumPoverN, double &alpha, double eq_tol, double alpha0) {
    double inizio = 0.2;
    double fattore = 2.0;
    bool sale = true;
    tolerance tolleranza = eq_tol;
    boost::uintmax_t massimo = 1000;
    std::pair<double, double> result = boost::math::tools::bracket_and_solve_root(std::bind(dsmm::eqforalpha,std::placeholders::_1,p,M,N,sumPoverN), inizio, fattore, sale, tolleranza, massimo);
    alpha = 0.5*(result.first + result.second);
}

void dsmm::sumPoverN(double *pwise_dist, int M, int N, double neighbor_cutoff, double *p, double *sumPoverN) {
    double avgd = 0.0;
    double min;
    for(int n=0;n<N;n++){
        min = 1e7;
        for(int np=0;np<N;np++){
            if(n!=np){
                if(pwise_dist[np*N+n]<min){min=pwise_dist[np*N+n];}
            }
        }
        avgd += min;
    }

    avgd /= N;
    avgd *= neighbor_cutoff;
    
    int mN;
    int neighborN;
    for(int m=0;m<M;m++){
        mN = m*N;
        for(int n=0;n<N;n++){
            neighborN = 0;
            for(int np=0;np<N;np++){
                if(pwise_dist[n*N+np]<avgd){
                    sumPoverN[mN+n] += p[mN+np];
                    neighborN++;
                }
            }
            if(neighborN!=0){sumPoverN[mN+n] /= neighborN;}
        }
    }
}
