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
	double conv_epsilon, double eq_tol, int max_iter,
	double *pwise_dist, double *pwise_distYY,
	double *Gamma, double *CDE_term,
	double *w, double *F_t, double *wF_t, double *wF_t_sum,
	double *p, double *u, int *Match,
	double *hatP, double *hatPI_diag, double *hatPIG, double *hatPX, double *hatPIY,
	double *G, double *W, double *GW,
	double *sumPoverN, double *expAlphaSumPoverN) {

	/***Registers Y onto X via a nonrigid pointset registration based on a
	Student's t-distribution mixture model with Dirichlet-distribution priors
	via an expectation-maximization algorithm. This is the "naked" C++
	implementation. See the wrapped versions in Python and LabView for more
	user-friendly list of arguments that do not require preallocation of
	arrays.

	Ref:
	[1] doi:10.1371/journal.pone.0091381
	[2] doi:10.1038/s41598-018-26288-6

	In the comments, Eq. () refers to Ref. [1], while Eq. ()' to Ref. [2].

	Parameters
	----------
	X, Y: array of doubles
		Sets of points in D-dimensional space (Y gets moved onto X). These
		arrays are modified inside this function: if you need to keep the
		original ones, pass copies.
		Note: Should be contiguous row-major arrays, with indices
		[point, coordinate].
	M: integer
		Number of points in Y.
	N: integer
		Number of points in X.
	D: integer
		Number of dimensions in which X and Y live.
	beta: double
		Standard deviation of Gaussian smoothing filter. See equations in the
		references. E.g.: 2.0
	lambda: double
		Regularization parameter. See equations in the references. E.g.: 1.5
	neighbor_cutoff: double
		Multiple of the average nearest-neighbor distance within which points
		are considered neighbors. See equations in the references. E.g.: 10.0
	gamma0: double
		Initialization of the gamma_m parameters (degrees of freedom of the
		Student's t-distribution). See equations in the references. E.g.: 1.0
	conv_epsilon: double
		Relative error on the displacements of the points in Y at which the
		algorithm is considered at convergence. E.g.: 1e-3
	eq_tol: double
		Tolerance for convergence of the numerical solution of the equations
		for gamma_m and \\bar alpha. See equations in the references.
		E.g.: 1e-4
	pwise_dist[M,N] double, pwise_distYY[M,M] double,
	Gamma[M] double, CDE_term[M] double,
	w[M,N] double, F_t[M,N] double, wF_t[M,N] double, wF_t_sum[N] double,
	p[M,N] double, u[M,N] double, Match[M,N] int,
	hatP[M,N] double, hatPI_diag[M] double, hatPIG[M,M] double,
	hatPX[M,D] double, hatPIY[M,D] double,
	W[M,D] double, GW[M,D] double,
	sumPoverN[M,N] double, expAlphaSumPoverN[M,N] double: arrays of double of
	specified dimensions and type
		Preallocated arrays, so that the memory can be reused through
		executions and their content is available to the outside.
		See description below for the relevant ones. All can be passed
		empty/uninitialized, all are populated inside this function.
		The names reflect the names of the variables in the equations in the
		references.
	p: array of doubles
		p[m,n] is the posterior probability for the match of Y[m] to X[n].
	Match: array of int
		X[Match[m]] is the point in X to which Y[m] has been matched. The
		built-in criterion is that the maximum posterior probability p[m,:] for
		Y[m] has to be greater than 0.3 and that the distance between the
		matched points has to be smaller than twice the average distance
		between all the matched points. If a different criterion is needed,
		use p to calculate the matches.
	**/

	// "Normalize" in Vemuri's language
	// Y -= avg of Ys
	// Y /= max of Ys (after the subtraction above) [max of all dimensions!]
	double avg, max;
	max=0.0;
	for(int d=0;d<D;d++) {
		avg=0.0;
		for (int m=0;m<M;m++) {
			avg+=Y[m*D+d];
		}
		avg /= M;
		for(int m=0;m<M;m++) {
			Y[m*D+d] -= avg;
		}
		for(int m=0;m<M;m++) {
			if (max<abs(Y[m*D+d])) {max = abs(Y[m*D+d]);}
		}
	}
	if(max!=0.0){
		for(int m=0;m<M;m++) { for(int d=0;d<D;d++) {
			Y[m*D+d] /= max; 
		}}
	}

	// Do the same for X. This time store the parameters for final 
	// denormalization. Since Y is moved onto X, the parameters used to
	// normalize X will be used to denormalized both X and Y.
	double *AvgX = new double[D];
	double maxX = 0.0;
    max = 0.0;
    for(int d=0;d<D;d++){
        avg = 0.0;
        for(int n=0;n<N;n++){
            avg += X[n*D+d];
        }
        avg /= N;
		AvgX[d] = avg;
        for(int n=0;n<N;n++){
            X[n*D+d] -= avg;
        }
        for(int n=0;n<N;n++){
            if(max<abs(X[n*D+d])){max=abs(X[n*D+d]);}
        }
    }
		
    if(max!=0.0){
        for(int n=0;n<N;n++){ for(int d=0;d<D;d++){
            X[n*D+d] /= max;
			maxX = max;
        }}
	}
	else {
		maxX = 1.0;
	}
    
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrice;
    typedef Eigen::Map<Matrice> MatrixMap;
    
	// Attach Eigen::Matrix header to some of the arrays, via Eigen::Map.
    MatrixMap X_(X,N,D), Y_(Y,M,D);
    MatrixMap pwise_dist_(pwise_dist,M,N), pwise_distYY_(pwise_distYY,M,M);
    MatrixMap w_(w,M,N), F_t_(F_t,M,N), p_(p,M,N), u_(u,M,N);
    MatrixMap hatP_(hatP,M,N), hatPIG_(hatPIG,M,M), hatPX_(hatPX,M,D), hatPIY_(hatPIY,M,D); 
    MatrixMap G_(G,M,M), W_(W,M,D), GW_(GW,M,D);
    
	// Initialize w.
	double oneovermn = 1. / M / N;
	for (int mn=0; mn < M*N; mn++) {
		w[mn] = oneovermn;
	}

	// Initialize Gamma
	for (int m = 0; m < M; m++) {
		Gamma[m] = gamma0;
	}
    
    dsmm::pwise_dist2(Y,X,M,N,D,pwise_dist);
    dsmm::pwise_dist2_same(Y,M,D,pwise_distYY); 
    
	// Initialize sigma, the errors, and beta**2
    double sigma2 = pwise_dist_.sum()/(D*M*N);
    double regerror=pwise_dist_.sum(), regerror_old=pwise_dist_.sum(), relerror=1000.;
    double beta2 = pow(beta,2.0);

	// Initialize G
    double gtmp;
    for(int m=0;m<M;m++){
        for(int m2=m;m2<M;m2++){
            gtmp = dsmm::fastexp(-pwise_distYY[m*M+m2]*0.5/beta2);
            //gtmp = exp(-pwise_distYY[m*M+m2]*0.5/beta2);
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
        
        double goldpdhalves, c_term,p_sum;
        for(int m=0;m<M;m++){
            goldpdhalves = 0.5*(Gamma[m]+D);
            c_term = 0.0;
            p_sum = 0.0;
            for(int n=0;n<N;n++){
                c_term += p[m*N+n]*(log(u[m*N+n])-u[m*N+n]);
                p_sum += p[m*N+n];
            }
            c_term /= p_sum;
            
            CDE_term[m] = c_term-log(goldpdhalves)+boost::math::digamma(goldpdhalves);
            //dsmm::logmenodigamma(goldpdhalves);//+d_term+e_term; FIXME    //FIXME fast
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
        
        //FIXME whole two for loops
        for(int m=0;m<M;m++){
            for(int m2=m;m2<M;m2++){
                gtmp = dsmm::fastexp(-pwise_distYY[m*M+m2]*0.5/beta2);
                //gtmp = exp(-pwise_distYY[m*M+m2]*0.5/beta2);
                G[m*M+m2] = gtmp;
                G[m2*M+m] = gtmp;
            }
        }
        
        for(int m=0;m<M;m++){
            for(int m2=0;m2<M;m2++){
                hatPIG[m*M+m2] = hatPI_diag[m]*G[m*M+m2];
                if(m2==m){hatPIG[m*M+m2] += lambda*sigma2;}
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
              
        //Relative error to check for convergence
        regerror_old = regerror;
        regerror = pwise_dist_.sum();
        relerror = abs((regerror-regerror_old)/regerror_old);
        if(regerror_old==0.0){break;}
        
        iter++;
        if(iter>0){beta2 *= 0.99*0.99;lambda *= 0.9*0.9;}
        if(iter>max_iter){break;}
    }
    
    /**
	Find the matches.
	The built-in criterion is that the maximum posterior probability p[m,:] for
	Y[m] has to be greater than 0.3 and that the distance between the
	matched points has to be smaller than twice the average distance
	between all the matched points.**/
	//(maybe filtered on the pairs with enough confidence
    //otherwise you also have the distances between points that are far because
    //the correspondence is missing.
    
    // If loop was terminated correctly
    if(mentre==true){
        // Find average of minimum distance between any Y and X.
        double mindist;
        double avgmindist=0.0;
		
		for(int m=0;m<M;m++){
            mindist = pwise_dist[m*N];
            for(int n=0;n<N;n++){
                if(mindist>pwise_dist[m*N+n]){
                    mindist=pwise_dist[m*N+n];
                }
            }
            avgmindist += mindist;
        }
        avgmindist /= M;
        
		// Find the matches (see above for description of criterion)
        double maxp,sump; // maxp2
        int match_index;
        for(int m=0;m<M;m++){
            match_index = -1;
            maxp = 0.0;
            //maxp2 = 0.0;
            sump = 0.0;
            for(int n=0;n<N;n++){
                sump += p[m*N+n];
                if(maxp<p[m*N+n]){
                    //maxp2 = maxp;
                    maxp = p[m*N+n];
                    match_index = n;
                }
            }
            if(
            (pwise_dist[m*N+match_index]<2.*avgmindist && (maxp/sump)>(0.3)) ||
            (pwise_dist[m*N+match_index]<3.*avgmindist && (maxp/sump)>0.8) ||
            (pwise_dist[m*N+match_index]<1.5*avgmindist)           
            ){ // 
                Match[m] = match_index;
            } else {
			    Match[m] = -1;
		    }
        }
        
        // Look for double matches
        int32_t *counts = new int32_t[N];
        for(int n=0;n<N;n++){counts[n]=0;}        
        for(int m=0;m<M;m++){if(Match[m]>=0){counts[Match[m]] += 1;}}
        
        // Select one out of those double matches.
        int surviving = 0;
        for(int n=0;n<N;n++){
            if(counts[n]>1){
                maxp = 0.0;
                for(int m=0;m<M;m++){
                    surviving = m;
                    if(Match[m]==n){
                        //Set all the matches pointing to the double match to -1
                        Match[m] = -1;
                        //Find the most confident candidate (incl distance?)
                        if(p[m*N+n]>maxp){maxp=p[m*N+n];surviving=m;}
                    }
                }
                Match[surviving] = n;
            }
        }
        delete[] counts;
    
    // If loop was not terminated correctly        
    } else {
        for(int m=0;m<M;m++){
            Match[m] = -1;
        }
    }
	
	bool denormalize = true;
	if(denormalize) {	
	    // "Denormalize" in Vemuri's language. Since Y has been moved onto X,
	    // denormalize both with the parameters originally used to normalize X.
	    for(int n=0;n<N;n++){
		    for(int d=0;d<D;d++) {
			    X[n*D+d] *= maxX;
			    X[n*D+d] += AvgX[d];
		    }
	    }

	    for(int m=0;m<M;m++){
		    for(int d=0;d<D;d++) {
			    Y[m*D+d] *= maxX;
			    Y[m*D+d] += AvgX[d];
		    }
	    }
    }
	
	delete[] AvgX;

}
