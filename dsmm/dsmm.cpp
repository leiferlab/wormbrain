#include "dsmm.hpp"
#include "dsmm_utils.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <boost/math/special_functions/digamma.hpp>

void dsmm::_dsmm(double *X, double *Y, int M, int N, int D,
           double beta, double lambda, double neighbor_cutoff,
           double alpha, double gamma0,
           double conv_epsilon, double eq_tol,
           double *pwise_dist, double *pwise_distYY,
           double *Gamma, double *CDE_term,
           double *w, double *F_t, double *wF_t, double *wF_t_sum, 
           double *p, double *u, int *Match,
           double *hatP, double *hatPI_diag, double *hatPIG, double *hatPX, double *hatPIY,
           double *G, double *W, double *GW, 
           double *sumPoverN, double *expAlphaSumPoverN) { //Passing the allocated arrays, Eigen Matrix header added inside.
    // write docs
    
    //"Normalize" in Vemuri's language
    double avg,max;
    for(int d=0;d<D;d++){
        avg = 0.0;
        max = 0.0;
        for(int m=0;m<M;m++){
            avg += Y[m*D+d];
        }
        avg /= M;
        for(int m=0;m<M;m++){
            Y[m*D+d] -= avg;
        }
        for(int m=0;m<M;m++){
            if(max<abs(Y[m*D+d])){max=abs(Y[m*D+d]);}
        }
        if(max!=0.0){
            for(int m=0;m<M;m++){
                Y[m*D+d] /= max;
            }
        }
    }
    for(int d=0;d<D;d++){
        avg = 0.0;
        max = 0.0;
        for(int n=0;n<N;n++){
            avg += X[n*D+d];
        }
        avg /= N;
        for(int n=0;n<N;n++){
            X[n*D+d] -= avg;
        }
        for(int n=0;n<N;n++){
            if(max<abs(X[n*D+d])){max=abs(X[n*D+d]);}
        }
        if(max!=0.0){
            for(int n=0;n<N;n++){
                X[n*D+d] /= max;
            }
        }
    }
    
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrice;
    typedef Eigen::Map<Matrice> MatrixMap;
    
    MatrixMap X_(X,N,D), Y_(Y,M,D);
    MatrixMap pwise_dist_(pwise_dist,M,N), pwise_distYY_(pwise_distYY,M,M);
    //MatrixMap Gamma_(Gamma,M,1);
    MatrixMap w_(w,M,N), F_t_(F_t,M,N), p_(p,M,N), u_(u,M,N);
    MatrixMap hatP_(hatP,M,N), hatPIG_(hatPIG,M,M), hatPX_(hatPX,M,D), hatPIY_(hatPIY,M,D); 
    MatrixMap G_(G,M,M), W_(W,M,D), GW_(GW,M,D);
    //MatrixMap sumPoverN_(sumPoverN,M,N);
    
	double oneovermn = 1. / M / N;
	for (int mn=0; mn < M*N; mn++) {
		w[mn] = oneovermn;
	}

	for (int m = 0; m < M; m++) {
		Gamma[m] = gamma0;
	}
    
    dsmm::pwise_dist2(Y,X,M,N,D,pwise_dist);
    dsmm::pwise_dist2_same(Y,M,D,pwise_distYY); 
    
    double sigma2 = pwise_dist_.sum()/(D*M*N);
    double regerror=pwise_dist_.sum(), regerror_old=pwise_dist_.sum(), relerror=1000.;
    double beta2 = pow(beta,2.0);
    
    /**for(int mm=0;mm<M*M;mm++){
        G[mm] = dsmm::fastexp(-pwise_distYY[mm]*0.5/beta2);
        //G[mm] = exp(-pwise_distYY[mm]*0.5/beta2);
    }**/
    // FIXME If something goes wrong, these are the newly introduced updates.
    double gtmp;
    for(int m=0;m<M;m++){
        for(int m2=m;m2<M;m2++){
            gtmp = dsmm::fastexp(-pwise_distYY[m*M+m2]*0.5/beta2);
            G[m*M+m2] = gtmp;
            G[m2*M+m] = gtmp;
        }
    }
    
    int iter = 0;
    double aa=0.0, bb=0.0, tmpgd, alpha_old;
    bool mentre = true;
    while((relerror>conv_epsilon) && (mentre)) {
        regerror_old=pwise_dist_.sum();
        //Step3 (Eq. (5))
        dsmm::studt(pwise_dist,M,N,sigma2,Gamma,D,F_t);
        
        //////Step3:E-Step
        //////Eq. (17)'
        //Python code:
        //wF_t = w*F_t
        //p = wF_t / np.sum(wF_t,axis=0)[None,:]
        for(int mn=0;mn<M*N;mn++){ wF_t[mn] = w[mn]*F_t[mn];}
        for(int n=0;n<N;n++){
            wF_t_sum[n] = 0.0;
            for(int m=0;m<M;m++){wF_t_sum[n] += wF_t[m*N+n];}
        }
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                p[m*N+n] = wF_t[m*N+n] / wF_t_sum[n];
            }
        }

        //Eq. (16) and (21)'
        //Python code
        //u[:] = (Gamma[:,None] + D) / (Gamma[:,None] + pwise_dist/sigma2)
        for(int m=0;m<M;m++){
            tmpgd = Gamma[m]+D;
            for(int n=0;n<N;n++){
                //u[m*N+n] = (Gamma[m]+D) / (Gamma[m]+pwise_dist[m*N+n]/sigma2);
                u[m*N+n] = tmpgd / (Gamma[m]+pwise_dist[m*N+n]/sigma2);
            }
        }
        
        //Eq. (20)'
        dsmm::sumPoverN(pwise_distYY, M, N, neighbor_cutoff, p, sumPoverN);
        alpha_old = alpha;
        dsmm::solveforalpha(p,M,N,sumPoverN,alpha,eq_tol,alpha_old);
        
        //Step4:M-Step
        // Eq. (18)'
        //Python code
        //expAlphaSumPoverN = np.exp(alpha*sumPoverN)
        //w[:] = expAlphaSumPoverN*(1./np.sum(expAlphaSumPoverN,axis=0)[None,:])
        /**for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                expAlphaSumPoverN[m*N+n] = dsmm::fastexp(alpha*sumPoverN[m*N+n]); 
            }
        }**/
        for(int mn=0;mn<M*N;mn++){
            expAlphaSumPoverN[mn] = dsmm::fastexp(alpha*sumPoverN[mn]);
            //expAlphaSumPoverN[mn] = exp(alpha*sumPoverN[mn]);
        }
        
        double somma;
        for(int n=0;n<N;n++){
            somma = 0.0;
            for(int m=0;m<M;m++){somma += expAlphaSumPoverN[m*N+n];}
            somma = 1./somma;
            for(int m=0;m<M;m++){ w[m*N+n] = expAlphaSumPoverN[m*N+n]*somma;}
        }
        
        //Eq. (23)
        //Python code
        //Gammaoldpdhalves = np.absolute(0.5*(Gamma_old+D))
        //C_term = np.sum(p*(np.log(u)-u),axis=1)/np.sum(p,axis=1)
        //D_term = spdigamma(Gammaoldpdhalves)
        //E_term = -np.log(Gammaoldpdhalves)
        //CDE_term = C_term + D_term + E_term
        
        double goldpdhalves, c_term,p_sum;//,d_term,e_term;
        for(int m=0;m<M;m++){
            goldpdhalves = 0.5*(Gamma[m]+D);
            c_term = 0.0;
            p_sum = 0.0;
            for(int n=0;n<N;n++){
                c_term += p[m*N+n]*(log(u[m*N+n])-u[m*N+n]);
                p_sum += p[m*N+n];
            }
            c_term /= p_sum;
            //d_term = dsmm::digamma(goldpdhalves);
            //e_term = -log(goldpdhalves);
            
            CDE_term[m] = c_term-dsmm::logmenodigamma(goldpdhalves);//+d_term+e_term; FIXME
        }
        
        dsmm::solveforgamma(CDE_term,M,Gamma,eq_tol);
        
        //Eq. (26)
        //Python code
        //hatP[:] = p*u
        //hatPI_diag[:] = np.sum(hatP,axis=1)
        //G[:] = np.exp(-0.5/beta2*pwise_distYY)
        //hatPIG[:] = hatPI_diag[:,None]*G #hatPIG+lambda*sigma2*Identity
        //hatPIY = hatPI_diag[:,None]*Y
        for(int m=0;m<M;m++){
            hatPI_diag[m] = 0.0;
            for(int n=0;n<N;n++){
                hatP[m*N+n] = p[m*N+n]*u[m*N+n];
                hatPI_diag[m] += hatP[m*N+n];
            }
        }
        /***********for(int m=0;m<M;m++){
            hatPI_diag[m] = 0.0;
            for(int n=0;n<N;n++){
                hatPI_diag[m] += hatP[m*N+n];
            }
        }
        for(int m=0;m<M;m++){
            for(int m2=0;m2<M;m2++){
                G[m*M+m2] = exp(-0.5/beta2*pwise_distYY[m*M+m2]);
            }
        }*************/
        
        //FIXME whole two for loops
        for(int m=0;m<M;m++){
            for(int m2=m;m2<M;m2++){
                gtmp = dsmm::fastexp(-pwise_distYY[m*M+m2]*0.5/beta2);
                G[m*M+m2] = gtmp;
                G[m2*M+m] = gtmp;
            }
        }
        
        for(int m=0;m<M;m++){
            for(int m2=0;m2<M;m2++){
                //G[m*M+m2] = dsmm::fastexp(-0.5/beta2*pwise_distYY[m*M+m2]); //FIXME
                //G[m*M+m2] = exp(-0.5/beta2*pwise_distYY[m*M+m2]);
                hatPIG[m*M+m2] = hatPI_diag[m]*G[m*M+m2];
                if(m2==m){ hatPIG[m*M+m2] += lambda*sigma2;}  
            }
        }
        
        
        
        for(int m=0;m<M;m++){
            for(int d=0;d<D;d++){hatPIY[m*D+d] = hatPI_diag[m]*Y[m*D+d];}
        }
        
        //Python code
        //hatPX[:] = np.dot(hatP,X)
        hatPX_.noalias() = hatP_*X_; 
        
        //Python code
        //#hatPIG+lambda*sigma2*Identity done above
        //A = np.linalg.inv(hatPIG+lambda*sigma2*I1_) 
        //B = hatPX - hatPIY
        //W = np.dot(A,B)
        W_.noalias() = hatPIG_.inverse()*(hatPX_-hatPIY_); 
        
        //Step5 moved here to optimize
        Y_.noalias() += G_*W_;
        //Y_.noalias() += G_*(hatPIG_.inverse()*(hatP_*X_-hatPIY_));
        
        dsmm::pwise_dist2(Y,X,M,N,D,pwise_dist);
        dsmm::pwise_dist2_same(Y,M,D,pwise_distYY);
        
        //Back to step 4
        //Eq. (27)
        //Python code
        //AA = np.sum(hatP*pwise_dist)
        //BB = D*np.sum(hatP)
        //sigma2 = AA*(1./BB)
        aa = 0.0;
        bb = 0.0;
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                aa += hatP[m*N+n]*pwise_dist[m*N+n];
                bb += hatP[m*N+n];
            }
        }
        bb *= D;
        sigma2 = aa/bb;
        //sigma2 = (hatP_.array()*pwise_dist_.array()).sum()/hatP_.sum();
        //sigma2 /= D; 
        
        
        //Relative error to check for convergence
        regerror_old = regerror;
        regerror = pwise_dist_.sum();
        relerror = abs((regerror-regerror_old)/regerror_old);
        if(regerror_old==0.0){mentre=false;}
        /**if(false){
            std::cout<<"F_t "<<F_t[0]<<"\n";
            std::cout<<"p "<<p[0]<<"\n";
            std::cout<<"u "<<u[0]<<"\n";
            std::cout<<"sumPoverN "<<sumPoverN[0]<<"\n";
            std::cout<<"alpha "<<alpha<<"\n";
            std::cout<<"CDE "<<CDE_term[0]<<"\n";
            std::cout<<"Gamma "<<Gamma[0]<<"\n";
            std::cout<<"W "<<W[0]<<"\n";
            std::cout<<"Y "<<Y[0]<<"\n";
            
            std::cout<<"sigma "<<sigma2<<"\n";
            //mentre=false;
        }**/
        iter++;
    }
    //std::cout<<iter<<"\n";
    
    // Find the closest matches (the posteriors sometimes go bad when there
    // are a lot of outliers).
    // FIXME This doesn't work really, because sometimes the match extracted via
    // the posteriors is reasonable but is not the closest match. You
    // practically need to run the nearest neighbor matching that you have
    // in Python, so that you take the argmax for the posterior if the
    // distance to that match is within a threshold from the median/average
    // first neighbor distance (maybe filtered on the pairs with enough confidence
    // otherwise you also have the distances between points that are far because
    // the correspondence is missing.
    
    if(mentre==true){
        // Find average of minimum distance in pwise_distYY
        double mindist;
        double avgmindist=0.0;
        for(int m=0;m<M;m++){
            mindist = pwise_distYY[m*M];
            for(int m2=0;m2<M;m2++){
                if(mindist>pwise_distYY[m*M+m2] && m2!=m){
                    mindist=pwise_distYY[m*M+m2];
                }
            }
            avgmindist += mindist;
        }
        avgmindist /= M;
        
        double maxp;
        int match_index;
        for(int m=0;m<M;m++){
            match_index = -1;
            maxp = 0.0;
            for(int n=0;n<N;n++){
                if(maxp<p[m*N+n]){
                    maxp = p[m*N+n];
                    match_index = n;
                }
            }
            if(pwise_dist[m*N+match_index]<2.*avgmindist && maxp>0.5){
                Match[m] = match_index;
            }
        }
    } else {
        for(int m=0;m<M;m++){
            Match[m] = -10.0;
        }
    }
}
