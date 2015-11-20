#include <iostream>

//#include "GCS.h"
#include "SubSystem.h"
#include "Util.h"
#include <algorithm>
#include <cfloat>
#include <limits>

#include "qp_eq.h"

//#define OLD_FULLPIVLU // This enforces old legacy FullPivLU::compute method
#define EIGEN_3_3_OR_BEYOND // if you are using eigen 3.3 or higher this must be set, if you are using eigen 3.2 this must be unset.


#ifdef OLD_FULLPIVLU

namespace Eigen {
    
    typedef Matrix<double,-1,-1,0,-1,-1> MatrixdType;
    
    #if !defined(EIGEN_3_3_OR_BEYOND)
    template<>
    FullPivLU<MatrixdType>& FullPivLU<MatrixdType>::compute(const MatrixdType& matrix)
    #else
    template<>
    template<>
    FullPivLU<MatrixdType>& FullPivLU<MatrixdType>::compute(const EigenBase<MatrixdType>& matrix)
    #endif
    {
        m_isInitialized = true;
        
        #if !defined(EIGEN_3_3_OR_BEYOND)  
        m_lu = matrix;
        #else
        m_lu = matrix.derived();
        #endif
        
        const Index size = m_lu.diagonalSize();
        const Index rows = matrix.rows();
        const Index cols = matrix.cols();
        
        // will store the transpositions, before we accumulate them at the end.
        // can't accumulate on-the-fly because that will be done in reverse order for the rows.
        m_rowsTranspositions.resize(matrix.rows());
        m_colsTranspositions.resize(matrix.cols());
        Index number_of_transpositions = 0; // number of NONTRIVIAL transpositions, i.e. m_rowsTranspositions[i]!=i
        
        m_nonzero_pivots = size; // the generic case is that in which all pivots are nonzero (invertible case)
        m_maxpivot = RealScalar(0);
        RealScalar cutoff(0);
        
        for(Index k = 0; k < size; ++k)
        {
            // First, we need to find the pivot.
            
            // biggest coefficient in the remaining bottom-right corner (starting at row k, col k)
            Index row_of_biggest_in_corner, col_of_biggest_in_corner;
            RealScalar biggest_in_corner;
            biggest_in_corner = m_lu.bottomRightCorner(rows-k, cols-k)
            .cwiseAbs()
            .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
            row_of_biggest_in_corner += k; // correct the values! since they were computed in the corner,
            col_of_biggest_in_corner += k; // need to add k to them.
            
            // when k==0, biggest_in_corner is the biggest coeff absolute value in the original matrix
            if(k == 0) cutoff = biggest_in_corner * NumTraits<Scalar>::epsilon();
            
            // if the pivot (hence the corner) is "zero", terminate to avoid generating nan/inf values.
            // Notice that using an exact comparison (biggest_in_corner==0) here, as Golub-van Loan do in
            // their pseudo-code, results in numerical instability! The cutoff here has been validated
            // by running the unit test 'lu' with many repetitions.
            if(biggest_in_corner < cutoff)
            {
                // before exiting, make sure to initialize the still uninitialized transpositions
                // in a sane state without destroying what we already have.
                m_nonzero_pivots = k;
                for(Index i = k; i < size; ++i)
                {
                    m_rowsTranspositions.coeffRef(i) = i;
                    m_colsTranspositions.coeffRef(i) = i;
                }
                break;
            }
            
            if(biggest_in_corner > m_maxpivot) m_maxpivot = biggest_in_corner;
            
            // Now that we've found the pivot, we need to apply the row/col swaps to
            // bring it to the location (k,k).
            
            m_rowsTranspositions.coeffRef(k) = row_of_biggest_in_corner;
            m_colsTranspositions.coeffRef(k) = col_of_biggest_in_corner;
            if(k != row_of_biggest_in_corner) {
                m_lu.row(k).swap(m_lu.row(row_of_biggest_in_corner));
                ++number_of_transpositions;
            }
            if(k != col_of_biggest_in_corner) {
                m_lu.col(k).swap(m_lu.col(col_of_biggest_in_corner));
                ++number_of_transpositions;
            }
            
            // Now that the pivot is at the right location, we update the remaining
            // bottom-right corner by Gaussian elimination.
            
            if(k<rows-1)
                m_lu.col(k).tail(rows-k-1) /= m_lu.coeff(k,k);
            if(k<size-1)
                m_lu.block(k+1,k+1,rows-k-1,cols-k-1).noalias() -= m_lu.col(k).tail(rows-k-1) * m_lu.row(k).tail(cols-k-1);
        }
        
        // the main loop is over, we still have to accumulate the transpositions to find the
        // permutations P and Q
        
        m_p.setIdentity(rows);
        for(Index k = size-1; k >= 0; --k)
            m_p.applyTranspositionOnTheRight(k, m_rowsTranspositions.coeff(k));
        
        m_q.setIdentity(cols);
        for(Index k = 0; k < size; ++k)
            m_q.applyTranspositionOnTheRight(k, m_colsTranspositions.coeff(k));
        
        m_det_pq = (number_of_transpositions%2) ? -1 : 1;
        return *this;
    }
    
} // Eigen

#endif

namespace GCS
{
    ///////////////////////////////////////
    // Other BFGS Solver parameters
    ///////////////////////////////////////
    #define XconvergenceRough 1e-8
    #define smallF            1e-20
    
    ///////////////////////////////////////
    // Solver
    ///////////////////////////////////////
    
    enum SolveStatus {
        Success = 0,        // Found a solution zeroing the error function
        Converged = 1,      // Found a solution minimizing the error function
        Failed = 2,         // Failed to find any solution
        SuccessfulSolutionInvalid = 3, // This is a solution where the solver succeeded, but the resulting geometry is OCE-invalid
    };
    
    enum Algorithm {
        BFGS = 0,
        LevenbergMarquardt = 1,
        DogLeg = 2
    };
    
    enum QRAlgorithm {
        EigenDenseQR = 0,
        EigenSparseQR = 1
    };
    
    enum DebugMode {
        NoDebug = 0,
        Minimal = 1,
        IterationLevel = 2
    };
}

using namespace GCS;

int solve_LM(GCS::SubSystem* subsys, bool isRedundantsolving);
int solve_DL(GCS::SubSystem* subsys, bool isRedundantsolving);

int maxIter=100;
bool sketchSizeMultiplier=true; // if true note that the total number of iterations allowed is MaxIterations *xLength
double convergence=1e-10;
double LM_eps=1E-10;
double LM_eps1=1E-80;          
double LM_tau=1E-3;
double DL_tolg=1E-80;
double DL_tolx=1E-80;          
double DL_tolf=1E-10;

int solve_LM(GCS::SubSystem* subsys, bool isRedundantsolving)
{
    int xsize = subsys->pSize();
    int csize = subsys->cSize();
    
    if (xsize == 0)
        return GCS::Success;
    
    Eigen::VectorXd e(csize), e_new(csize); // vector of all function errors (every constraint is one function)
    Eigen::MatrixXd J(csize, xsize);        // Jacobi of the subsystem
    Eigen::MatrixXd A(xsize, xsize);
    Eigen::VectorXd x(xsize), h(xsize), x_new(xsize), g(xsize), diag_A(xsize);
    
    subsys->redirectParams();
    
    subsys->getParams(x);
    subsys->calcResidual(e);
    e*=-1;
    
    int maxIterNumber = maxIter;
    
    double divergingLim = 1e6*e.squaredNorm() + 1e12;
    
    double eps=(LM_eps);
    double eps1=(LM_eps1);
    double tau=(LM_tau);
    
        
    std::cout << "LM: eps: "          << eps
    << ", eps1: "           << eps1
    << ", tau: "            << tau
    << ", convergence: "    << (convergence)
    << ", xsize: "          << xsize                
    << ", maxIter: "        << maxIterNumber  << "\n";
        
    
    double nu=2, mu=0;
    int iter=0, stop=0;
    for (iter=0; iter < maxIterNumber && !stop; ++iter) {
        
        // check error
        double err=e.squaredNorm();
        if (err <= eps) { // error is small, Success
            stop = 1;
            break;
        }
        else if (err > divergingLim || err != err) { // check for diverging and NaN
            stop = 6;
            break;
        }
        
        // J^T J, J^T e
        subsys->calcJacobi(J);;
        
        A = J.transpose()*J;
        g = J.transpose()*e;
        
        // Compute ||J^T e||_inf
        double g_inf = g.lpNorm<Eigen::Infinity>();
        diag_A = A.diagonal(); // save diagonal entries so that augmentation can be later canceled
        
        // check for convergence
        if (g_inf <= eps1) {
            stop = 2;
            break;
        }
        
        // compute initial damping factor
        if (iter == 0)
            mu = tau * diag_A.lpNorm<Eigen::Infinity>();
        
        double h_norm;
        // determine increment using adaptive damping
        int k=0;
        while (k < 50) {
            // augment normal equations A = A+uI
            for (int i=0; i < xsize; ++i)
                A(i,i) += mu;
            
            //solve augmented functions A*h=-g
            h = A.fullPivLu().solve(g);
            double rel_error = (A*h - g).norm() / g.norm();
            
            // check if solving works
            if (rel_error < 1e-5) {
                
                // restrict h according to maxStep
                double scale = subsys->maxStep(h);
                if (scale < 1.)
                    h *= scale;
                
                // compute par's new estimate and ||d_par||^2
                x_new = x + h;
                h_norm = h.squaredNorm();
                
                if (h_norm <= eps1*eps1*x.norm()) { // relative change in p is small, stop
                    stop = 3;
                    break;
                }
                else if (h_norm >= (x.norm()+eps1)/(DBL_EPSILON*DBL_EPSILON)) { // almost singular
                    stop = 4;
                    break;
                }
                
                subsys->setParams(x_new);
                subsys->calcResidual(e_new);
                e_new *= -1;
                
                double dF = e.squaredNorm() - e_new.squaredNorm();
                double dL = h.dot(mu*h+g);
                
                if (dF>0. && dL>0.) { // reduction in error, increment is accepted
                    double tmp=2*dF/dL-1.;
                    mu *= std::max(1./3., 1.-tmp*tmp*tmp);
                    nu=2;
                    
                    // update par's estimate
                    x = x_new;
                    e = e_new;
                    break;
                }
            }
            
            // if this point is reached, either the linear system could not be solved or
            // the error did not reduce; in any case, the increment must be rejected
            
            mu*=nu;
            nu*=2.0;
            for (int i=0; i < xsize; ++i) // restore diagonal J^T J entries
                A(i,i) = diag_A(i);
            
            k++;
        }
        if (k > 50) {
            stop = 7;
            break;
        }
        
        std::stringstream stream;
        // Iteration: 1, residual: 1e-3, tolg: 1e-5, tolx: 1e-3
        
        std::cout  << "LM, Iteration: "            << iter
        << ", err(eps): "               << err
        << ", g_inf(eps1): "            << g_inf
        << ", h_norm: "                 << h_norm << "\n";
            
    }
    
    if (iter >= maxIterNumber)
        stop = 5;
    
    subsys->revertParams();
    
    return (stop == 1) ? GCS::Success : GCS::Failed;
}


int solve_DL(GCS::SubSystem* subsys, bool isRedundantsolving)
{
    double tolg=(DL_tolg);
    double tolx=(DL_tolx);
    double tolf=(DL_tolf);
    
    int xsize = subsys->pSize();
    int csize = subsys->cSize();
    
    if (xsize == 0)
        return GCS::Success;
    
    int maxIterNumber = (maxIter);
    
        
    std::cout  << "DL: tolg: "         << tolg
    << ", tolx: "           << tolx
    << ", tolf: "           << tolf
    << ", convergence: "    << (convergence)
    << ", xsize: "          << xsize
    << ", csize: "          << csize
    << ", maxIter: "        << maxIterNumber  << "\n";
        
      
    
    Eigen::VectorXd x(xsize), x_new(xsize);
    Eigen::VectorXd fx(csize), fx_new(csize);
    Eigen::MatrixXd Jx(csize, xsize), Jx_new(csize, xsize);
    Eigen::VectorXd g(xsize), h_sd(xsize), h_gn(xsize), h_dl(xsize);
    
    subsys->redirectParams();
    
    double err;
    subsys->getParams(x);
    subsys->calcResidual(fx, err);
    subsys->calcJacobi(Jx);
    
    g = Jx.transpose()*(-fx);
    
    // get the infinity norm fx_inf and g_inf
    double g_inf = g.lpNorm<Eigen::Infinity>();
    double fx_inf = fx.lpNorm<Eigen::Infinity>();
    
    double divergingLim = 1e6*err + 1e12;
    
    double delta=0.1;
    double alpha=0.;
    double nu=2.;
    int iter=0, stop=0, reduce=0;
    while (!stop) {
        
        // check if finished
        if (fx_inf <= tolf) // Success
            stop = 1;
        else if (g_inf <= tolg)
            stop = 2;
        else if (delta <= tolx*(tolx + x.norm()))
            stop = 2;
        else if (iter >= maxIterNumber)
            stop = 4;
        else if (err > divergingLim || err != err) { // check for diverging and NaN
            stop = 6;
        }
        else {
            // get the steepest descent direction
            alpha = g.squaredNorm()/(Jx*g).squaredNorm();
            h_sd  = alpha*g;
            
            // get the gauss-newton step
            h_gn = Jx.fullPivLu().solve(-fx);
            double rel_error = (Jx*h_gn + fx).norm() / fx.norm();
            if (rel_error > 1e15)
                break;
            
            // compute the dogleg step
            if (h_gn.norm() < delta) {
                h_dl = h_gn;
                if  (h_dl.norm() <= tolx*(tolx + x.norm())) {
                    stop = 5;
                    break;
                }
            }
            else if (alpha*g.norm() >= delta) {
                h_dl = (delta/(alpha*g.norm()))*h_sd;
            }
            else {
                //compute beta
                double beta = 0;
                Eigen::VectorXd b = h_gn - h_sd;
                double bb = (b.transpose()*b).norm();
                double gb = (h_sd.transpose()*b).norm();
                double c = (delta + h_sd.norm())*(delta - h_sd.norm());
                
                if (gb > 0)
                    beta = c / (gb + sqrt(gb * gb + c * bb));
                else
                    beta = (sqrt(gb * gb + c * bb) - gb)/bb;
                
                // and update h_dl and dL with beta
                h_dl = h_sd + beta*b;
            }
        }
        
        // see if we are already finished
        if (stop)
            break;
        
        // it didn't work in some tests
        //        // restrict h_dl according to maxStep
        //        double scale = subsys->maxStep(h_dl);
        //        if (scale < 1.)
        //            h_dl *= scale;
        
        // get the new values
        double err_new;
        x_new = x + h_dl;
        subsys->setParams(x_new);
        subsys->calcResidual(fx_new, err_new);
        subsys->calcJacobi(Jx_new);
        
        // calculate the linear model and the update ratio
        double dL = err - 0.5*(fx + Jx*h_dl).squaredNorm();
        double dF = err - err_new;
        double rho = dL/dF;
        
        if (dF > 0 && dL > 0) {
            x  = x_new;
            Jx = Jx_new;
            fx = fx_new;
            err = err_new;
            
            g = Jx.transpose()*(-fx);
            
            // get infinity norms
            g_inf = g.lpNorm<Eigen::Infinity>();
            fx_inf = fx.lpNorm<Eigen::Infinity>();
        }
        else
            rho = -1;
        
        // update delta
        if (fabs(rho-1.) < 0.2 && h_dl.norm() > delta/3. && reduce <= 0) {
            delta = 3*delta;
            nu = 2;
            reduce = 0;
        }
        else if (rho < 0.25) {
            delta = delta/nu;
            nu = 2*nu;
            reduce = 2;
        }
        else
            reduce--;
                    
        std::cout  << "DL, Iteration: "        << iter
        << ", fx_inf(tolf): "       << fx_inf
        << ", g_inf(tolg): "        << g_inf
        << ", delta(f(tolx)): "     << delta
        << ", err(divergingLim): "  << err  << "\n";
            
 
        
        // count this iteration and start again
        iter++;
    }
    
    subsys->revertParams();
    
        
    std::cout  << "DL: stopcode: "     << stop << ((stop == 1) ? ", Success" : ", Failed") << "\n";
          
    
    return (stop == 1) ? GCS::Success : GCS::Failed;
}

int main(int argc, char **argv) {
    std::cout << "This is a demontration of an Eigen algorithm relevant to FreeCAD" << std::endl << std::endl;
    
    std::cout << "Populating values for the relevant Subsystem" << std::endl;
    
    GCS::VEC_pD plist_;
    std::vector<GCS::Constraint *> clist_;
    GCS::VEC_pD plistsub_;
    GCS::VEC_pD clist_params_;
    plist_.push_back(new double(-100)); // 0 address: 0x4208ea0
    plist_.push_back(new double(-4e-11)); // 1 address: 0x42dce00
    plist_.push_back(new double(10)); // 2 address: 0x42bd340
    plist_.push_back(new double(-37.5)); // 3 address: 0x4161500
    plist_.push_back(new double(1.77e-10)); // 4 address: 0x41e7780
    plist_.push_back(new double(10)); // 5 address: 0x4205700
    plist_.push_back(new double(37.5)); // 6 address: 0x41efe80
    plist_.push_back(new double(-2.9e-11)); // 7 address: 0x4202700
    plist_.push_back(new double(10)); // 8 address: 0x42c9940
    plist_.push_back(new double(100)); // 9 address: 0x42dc760
    plist_.push_back(new double(-0)); // 10 address: 0x41f0570
    plist_.push_back(new double(10)); // 11 address: 0x41f0830
    plist_.push_back(new double(-82.6389)); // 12 address: 0x4096550
    plist_.push_back(new double(9.92934)); // 13 address: 0x41f6810
    plist_.push_back(new double(-107.516)); // 14 address: 0x41f6830
    plist_.push_back(new double(-18.5339)); // 15 address: 0x41f6850
    plist_.push_back(new double(-100)); // 16 address: 0x41f6870
    plist_.push_back(new double(-4e-11)); // 17 address: 0x40d90c0
    plist_.push_back(new double(20)); // 18 address: 0x40d90e0
    plist_.push_back(new double(0.519524)); // 19 address: 0x4191ef0
    plist_.push_back(new double(4.32712)); // 20 address: 0x4191f10
    plist_.push_back(new double(-82.6389)); // 21 address: 0x41c8050
    plist_.push_back(new double(9.92934)); // 22 address: 0x41ee8f0
    plist_.push_back(new double(-54.8611)); // 23 address: 0x41bfa70
    plist_.push_back(new double(9.92934)); // 24 address: 0x41ee910
    plist_.push_back(new double(-68.75)); // 25 address: 0x41b87f0
    plist_.push_back(new double(17.8728)); // 26 address: 0x41b8810
    plist_.push_back(new double(16)); // 27 address: 0x4178970
    plist_.push_back(new double(3.66112)); // 28 address: 0x4178990
    plist_.push_back(new double(5.76366)); // 29 address: 0x10cc6f0
    plist_.push_back(new double(-18.7272)); // 30 address: 0x419fce0
    plist_.push_back(new double(6.89794)); // 31 address: 0x419fda0
    plist_.push_back(new double(-54.8611)); // 32 address: 0x419fd00
    plist_.push_back(new double(9.92934)); // 33 address: 0x405f4f0
    plist_.push_back(new double(-37.5)); // 34 address: 0x405f510
    plist_.push_back(new double(1.77e-10)); // 35 address: 0x41592f0
    plist_.push_back(new double(20)); // 36 address: 0x4159310
    plist_.push_back(new double(0.352129)); // 37 address: 0x417a5c0
    plist_.push_back(new double(2.62207)); // 38 address: 0x417a5e0
    plist_.push_back(new double(-8.84372)); // 39 address: 0x4151bf0
    plist_.push_back(new double(1e-12)); // 40 address: 0x414c1f0
    plist_.push_back(new double(8.59986)); // 41 address: 0x4162f90
    plist_.push_back(new double(-1e-12)); // 42 address: 0x41e1170
    plist_.push_back(new double(-18.7272)); // 43 address: 0x42d6550
    plist_.push_back(new double(6.89794)); // 44 address: 0x3d7e640
    plist_.push_back(new double(-8.84372)); // 45 address: 0x3d6b910
    plist_.push_back(new double(2.00018e-12)); // 46 address: 0x42cb0d0
    plist_.push_back(new double(-8.84372)); // 47 address: 0x4186710
    plist_.push_back(new double(10.5296)); // 48 address: 0x40b3790
    plist_.push_back(new double(10.5296)); // 49 address: 0x40b5190
    plist_.push_back(new double(3.49372)); // 50 address: 0x4209ba0
    plist_.push_back(new double(4.71239)); // 51 address: 0x40b21f0
    plist_.push_back(new double(8.59986)); // 52 address: 0x41fd260
    plist_.push_back(new double(-1.00009e-12)); // 53 address: 0x41f69f0
    plist_.push_back(new double(18.7826)); // 54 address: 0x41fd280
    plist_.push_back(new double(7.04682)); // 55 address: 0x42caa80
    plist_.push_back(new double(8.59986)); // 56 address: 0x42caaa0
    plist_.push_back(new double(10.8804)); // 57 address: 0x42caac0
    plist_.push_back(new double(10.8804)); // 58 address: 0x42caae0
    plist_.push_back(new double(4.71239)); // 59 address: 0x42cab00
    plist_.push_back(new double(5.92311)); // 60 address: 0x42cab20
    plist_.push_back(new double(54.8611)); // 61 address: 0x42db6c0
    plist_.push_back(new double(9.92934)); // 62 address: 0x42db6a0
    plist_.push_back(new double(18.7826)); // 63 address: 0x41febe0
    plist_.push_back(new double(7.04682)); // 64 address: 0x41fec00
    plist_.push_back(new double(37.5)); // 65 address: 0x41fea60
    plist_.push_back(new double(-2.9e-11)); // 66 address: 0x42d4a10
    plist_.push_back(new double(20)); // 67 address: 0x42d4a30
    plist_.push_back(new double(0.519524)); // 68 address: 0x41f6a10
    plist_.push_back(new double(2.78152)); // 69 address: 0x41f6a30
    plist_.push_back(new double(54.8611)); // 70 address: 0x42dc090
    plist_.push_back(new double(9.92934)); // 71 address: 0x41f48d0
    plist_.push_back(new double(82.6389)); // 72 address: 0x41f48f0
    plist_.push_back(new double(9.92934)); // 73 address: 0x41f4910
    plist_.push_back(new double(68.75)); // 74 address: 0x41f4930
    plist_.push_back(new double(17.8728)); // 75 address: 0x41f4950
    plist_.push_back(new double(16)); // 76 address: 0x41f4970
    plist_.push_back(new double(3.66112)); // 77 address: 0x41f4990
    plist_.push_back(new double(5.76366)); // 78 address: 0x41f49b0
    plist_.push_back(new double(113.295)); // 79 address: 0x42dacf0
    plist_.push_back(new double(-14.9418)); // 80 address: 0x42dacd0
    plist_.push_back(new double(82.6389)); // 81 address: 0x42db0d0
    plist_.push_back(new double(9.92934)); // 82 address: 0x42db0f0
    plist_.push_back(new double(100)); // 83 address: 0x42db110
    plist_.push_back(new double(-0)); // 84 address: 0x40b5b20
    plist_.push_back(new double(20)); // 85 address: 0x40b5b40
    plist_.push_back(new double(5.43952)); // 86 address: 0x41fe2b0
    plist_.push_back(new double(8.90525)); // 87 address: 0x41fe2d0
    plist_.push_back(new double(70)); // 88 address: 0x42d73d0
    plist_.push_back(new double(-20)); // 89 address: 0x42d73b0
    plist_.push_back(new double(15)); // 90 address: 0x42d7730
    plist_.push_back(new double(-70)); // 91 address: 0x41c6ce0
    plist_.push_back(new double(-20)); // 92 address: 0x4186b50
    plist_.push_back(new double(15)); // 93 address: 0x4186b70
    plist_.push_back(new double(-0)); // 94 address: 0x40b6590
    plist_.push_back(new double(-40)); // 95 address: 0x42d78e0
    plist_.push_back(new double(12)); // 96 address: 0x42d7900
    plist_.push_back(new double(-91.729)); // 97 address: 0x41fc060
    plist_.push_back(new double(-32.3633)); // 98 address: 0x41fc080
    plist_.push_back(new double(-54.2208)); // 99 address: 0x41fc0a0
    plist_.push_back(new double(-39.3912)); // 100 address: 0x41fc0c0
    plist_.push_back(new double(-70)); // 101 address: 0x41fc0e0
    plist_.push_back(new double(-20)); // 102 address: 0x41865e0
    plist_.push_back(new double(25)); // 103 address: 0x4186600
    plist_.push_back(new double(3.65889)); // 104 address: 0x40b67a0
    plist_.push_back(new double(5.39545)); // 105 address: 0x40b67c0
    plist_.push_back(new double(-91.729)); // 106 address: 0x42d8910
    plist_.push_back(new double(-32.3633)); // 107 address: 0x42d88f0
    plist_.push_back(new double(-107.516)); // 108 address: 0x42d8d70
    plist_.push_back(new double(-18.5339)); // 109 address: 0x42d8d90
    plist_.push_back(new double(-119.542)); // 110 address: 0x42d8db0
    plist_.push_back(new double(-48.1882)); // 111 address: 0x42d8dd0
    plist_.push_back(new double(32)); // 112 address: 0x42d8df0
    plist_.push_back(new double(0.517294)); // 113 address: 0x42d8e10
    plist_.push_back(new double(1.18553)); // 114 address: 0x42d8e30
    plist_.push_back(new double(-54.2208)); // 115 address: 0x41dd2c0
    plist_.push_back(new double(-39.3912)); // 116 address: 0x41dd2a0
    plist_.push_back(new double(-24.8766)); // 117 address: 0x42e9fe0
    plist_.push_back(new double(-15.5129)); // 118 address: 0x42ea000
    plist_.push_back(new double(-19.3596)); // 119 address: 0x42ea150
    plist_.push_back(new double(-29.5499)); // 120 address: 0x42ea170
    plist_.push_back(new double(19.6015)); // 121 address: 0x42ea190
    plist_.push_back(new double(-49.9891)); // 122 address: 0x42ea1b0
    plist_.push_back(new double(-0)); // 123 address: 0x42ea1d0
    plist_.push_back(new double(-40)); // 124 address: 0x42ea1f0
    plist_.push_back(new double(22)); // 125 address: 0x42ea210
    plist_.push_back(new double(2.64662)); // 126 address: 0x42ea230
    plist_.push_back(new double(5.81188)); // 127 address: 0x42ea250
    plist_.push_back(new double(-19.3596)); // 128 address: 0x42ea940
    plist_.push_back(new double(-29.5499)); // 129 address: 0x42ea920
    plist_.push_back(new double(-11.1684)); // 130 address: 0x42eaa60
    plist_.push_back(new double(-14.3751)); // 131 address: 0x3d97780
    plist_.push_back(new double(-24.8766)); // 132 address: 0x4200df0
    plist_.push_back(new double(-15.5129)); // 133 address: 0x4200e10
    plist_.push_back(new double(-19.9546)); // 134 address: 0x4200e30
    plist_.push_back(new double(-9.6)); // 135 address: 0x4200e50
    plist_.push_back(new double(-37.5)); // 136 address: 0x4200e70
    plist_.push_back(new double(1.77e-10)); // 137 address: 0x4200e90
    plist_.push_back(new double(20)); // 138 address: 0x4200eb0
    plist_.push_back(new double(5.39545)); // 139 address: 0x4200ed0
    plist_.push_back(new double(5.78253)); // 140 address: 0x4200ef0
    plist_.push_back(new double(-11.1684)); // 141 address: 0x42eba20
    plist_.push_back(new double(-14.3751)); // 142 address: 0x42eba00
    plist_.push_back(new double(-19.9546)); // 143 address: 0x42ebe00
    plist_.push_back(new double(-9.6)); // 144 address: 0x42ebe20
    plist_.push_back(new double(-15.5683)); // 145 address: 0x42ebe40
    plist_.push_back(new double(-12)); // 146 address: 0x42ebe60
    plist_.push_back(new double(5)); // 147 address: 0x42ebe80
    plist_.push_back(new double(5.78821)); // 148 address: 0x42ebea0
    plist_.push_back(new double(8.92412)); // 149 address: 0x42ebec0
    plist_.push_back(new double(113.295)); // 150 address: 0x42ec600
    plist_.push_back(new double(-14.9418)); // 151 address: 0x42ec5e0
    plist_.push_back(new double(86.6181)); // 152 address: 0x42eca00
    plist_.push_back(new double(-38.6772)); // 153 address: 0x42eca20
    plist_.push_back(new double(55.9379)); // 154 address: 0x42ecb10
    plist_.push_back(new double(-40.6702)); // 155 address: 0x42ecb30
    plist_.push_back(new double(86.6181)); // 156 address: 0x42ecb50
    plist_.push_back(new double(-38.6772)); // 157 address: 0x42ecb70
    plist_.push_back(new double(70)); // 158 address: 0x42ecb90
    plist_.push_back(new double(-20)); // 159 address: 0x42ecbb0
    plist_.push_back(new double(25)); // 160 address: 0x42ecbd0
    plist_.push_back(new double(4.115)); // 161 address: 0x42ecbf0
    plist_.push_back(new double(5.43951)); // 162 address: 0x42ecc10
    plist_.push_back(new double(55.9379)); // 163 address: 0x42ed300
    plist_.push_back(new double(-40.6702)); // 164 address: 0x42ed2e0
    plist_.push_back(new double(19.6015)); // 165 address: 0x42ed6e0
    plist_.push_back(new double(-49.9891)); // 166 address: 0x42ed700
    plist_.push_back(new double(41.8758)); // 167 address: 0x42ed720
    plist_.push_back(new double(-61.3404)); // 168 address: 0x42ed740
    plist_.push_back(new double(25)); // 169 address: 0x42ed760
    plist_.push_back(new double(0.973411)); // 170 address: 0x42ed780
    plist_.push_back(new double(2.67028)); // 171 address: 0x42ed7a0
    plistsub_.push_back(plist_[29]); // 0
    plistsub_.push_back(plist_[45]); // 1
    plistsub_.push_back(plist_[44]); // 2
    plistsub_.push_back(plist_[131]); // 3
    plistsub_.push_back(plist_[33]); // 4
    plistsub_.push_back(plist_[34]); // 5
    plistsub_.push_back(plist_[12]); // 6
    plistsub_.push_back(plist_[51]); // 7
    plistsub_.push_back(plist_[48]); // 8
    plistsub_.push_back(plist_[49]); // 9
    plistsub_.push_back(plist_[84]); // 10
    plistsub_.push_back(plist_[85]); // 11
    plistsub_.push_back(plist_[94]); // 12
    plistsub_.push_back(plist_[104]); // 13
    plistsub_.push_back(plist_[105]); // 14
    plistsub_.push_back(plist_[17]); // 15
    plistsub_.push_back(plist_[18]); // 16
    plistsub_.push_back(plist_[40]); // 17
    plistsub_.push_back(plist_[39]); // 18
    plistsub_.push_back(plist_[35]); // 19
    plistsub_.push_back(plist_[36]); // 20
    plistsub_.push_back(plist_[3]); // 21
    plistsub_.push_back(plist_[41]); // 22
    plistsub_.push_back(plist_[27]); // 23
    plistsub_.push_back(plist_[28]); // 24
    plistsub_.push_back(plist_[37]); // 25
    plistsub_.push_back(plist_[38]); // 26
    plistsub_.push_back(plist_[102]); // 27
    plistsub_.push_back(plist_[103]); // 28
    plistsub_.push_back(plist_[47]); // 29
    plistsub_.push_back(plist_[92]); // 30
    plistsub_.push_back(plist_[93]); // 31
    plistsub_.push_back(plist_[19]); // 32
    plistsub_.push_back(plist_[20]); // 33
    plistsub_.push_back(plist_[30]); // 34
    plistsub_.push_back(plist_[32]); // 35
    plistsub_.push_back(plist_[31]); // 36
    plistsub_.push_back(plist_[25]); // 37
    plistsub_.push_back(plist_[26]); // 38
    plistsub_.push_back(plist_[23]); // 39
    plistsub_.push_back(plist_[91]); // 40
    plistsub_.push_back(plist_[21]); // 41
    plistsub_.push_back(plist_[116]); // 42
    plistsub_.push_back(plist_[115]); // 43
    plistsub_.push_back(plist_[42]); // 44
    plistsub_.push_back(plist_[4]); // 45
    plistsub_.push_back(plist_[22]); // 46
    plistsub_.push_back(plist_[24]); // 47
    plistsub_.push_back(plist_[6]); // 48
    plistsub_.push_back(plist_[10]); // 49
    plistsub_.push_back(plist_[11]); // 50
    plistsub_.push_back(plist_[71]); // 51
    plistsub_.push_back(plist_[72]); // 52
    plistsub_.push_back(plist_[73]); // 53
    plistsub_.push_back(plist_[74]); // 54
    plistsub_.push_back(plist_[75]); // 55
    plistsub_.push_back(plist_[76]); // 56
    plistsub_.push_back(plist_[77]); // 57
    plistsub_.push_back(plist_[78]); // 58
    plistsub_.push_back(plist_[13]); // 59
    plistsub_.push_back(plist_[14]); // 60
    plistsub_.push_back(plist_[15]); // 61
    plistsub_.push_back(plist_[16]); // 62
    plistsub_.push_back(plist_[53]); // 63
    plistsub_.push_back(plist_[68]); // 64
    plistsub_.push_back(plist_[69]); // 65
    plistsub_.push_back(plist_[97]); // 66
    plistsub_.push_back(plist_[98]); // 67
    plistsub_.push_back(plist_[99]); // 68
    plistsub_.push_back(plist_[100]); // 69
    plistsub_.push_back(plist_[101]); // 70
    plistsub_.push_back(plist_[52]); // 71
    plistsub_.push_back(plist_[54]); // 72
    plistsub_.push_back(plist_[86]); // 73
    plistsub_.push_back(plist_[87]); // 74
    plistsub_.push_back(plist_[65]); // 75
    plistsub_.push_back(plist_[63]); // 76
    plistsub_.push_back(plist_[64]); // 77
    plistsub_.push_back(plist_[132]); // 78
    plistsub_.push_back(plist_[133]); // 79
    plistsub_.push_back(plist_[134]); // 80
    plistsub_.push_back(plist_[135]); // 81
    plistsub_.push_back(plist_[136]); // 82
    plistsub_.push_back(plist_[137]); // 83
    plistsub_.push_back(plist_[138]); // 84
    plistsub_.push_back(plist_[139]); // 85
    plistsub_.push_back(plist_[140]); // 86
    plistsub_.push_back(plist_[7]); // 87
    plistsub_.push_back(plist_[5]); // 88
    plistsub_.push_back(plist_[0]); // 89
    plistsub_.push_back(plist_[50]); // 90
    plistsub_.push_back(plist_[2]); // 91
    plistsub_.push_back(plist_[8]); // 92
    plistsub_.push_back(plist_[55]); // 93
    plistsub_.push_back(plist_[56]); // 94
    plistsub_.push_back(plist_[57]); // 95
    plistsub_.push_back(plist_[58]); // 96
    plistsub_.push_back(plist_[59]); // 97
    plistsub_.push_back(plist_[60]); // 98
    plistsub_.push_back(plist_[46]); // 99
    plistsub_.push_back(plist_[66]); // 100
    plistsub_.push_back(plist_[67]); // 101
    plistsub_.push_back(plist_[43]); // 102
    plistsub_.push_back(plist_[89]); // 103
    plistsub_.push_back(plist_[88]); // 104
    plistsub_.push_back(plist_[90]); // 105
    plistsub_.push_back(plist_[95]); // 106
    plistsub_.push_back(plist_[96]); // 107
    plistsub_.push_back(plist_[107]); // 108
    plistsub_.push_back(plist_[106]); // 109
    plistsub_.push_back(plist_[108]); // 110
    plistsub_.push_back(plist_[109]); // 111
    plistsub_.push_back(plist_[110]); // 112
    plistsub_.push_back(plist_[111]); // 113
    plistsub_.push_back(plist_[112]); // 114
    plistsub_.push_back(plist_[113]); // 115
    plistsub_.push_back(plist_[114]); // 116
    plistsub_.push_back(plist_[80]); // 117
    plistsub_.push_back(plist_[79]); // 118
    plistsub_.push_back(plist_[81]); // 119
    plistsub_.push_back(plist_[82]); // 120
    plistsub_.push_back(plist_[83]); // 121
    plistsub_.push_back(plist_[62]); // 122
    plistsub_.push_back(plist_[61]); // 123
    plistsub_.push_back(plist_[70]); // 124
    plistsub_.push_back(plist_[9]); // 125
    plistsub_.push_back(plist_[1]); // 126
    plistsub_.push_back(plist_[117]); // 127
    plistsub_.push_back(plist_[118]); // 128
    plistsub_.push_back(plist_[119]); // 129
    plistsub_.push_back(plist_[120]); // 130
    plistsub_.push_back(plist_[121]); // 131
    plistsub_.push_back(plist_[122]); // 132
    plistsub_.push_back(plist_[123]); // 133
    plistsub_.push_back(plist_[124]); // 134
    plistsub_.push_back(plist_[125]); // 135
    plistsub_.push_back(plist_[126]); // 136
    plistsub_.push_back(plist_[127]); // 137
    plistsub_.push_back(plist_[129]); // 138
    plistsub_.push_back(plist_[128]); // 139
    plistsub_.push_back(plist_[130]); // 140
    plistsub_.push_back(plist_[142]); // 141
    plistsub_.push_back(plist_[141]); // 142
    plistsub_.push_back(plist_[143]); // 143
    plistsub_.push_back(plist_[144]); // 144
    plistsub_.push_back(plist_[145]); // 145
    plistsub_.push_back(plist_[146]); // 146
    plistsub_.push_back(plist_[147]); // 147
    plistsub_.push_back(plist_[148]); // 148
    plistsub_.push_back(plist_[149]); // 149
    plistsub_.push_back(plist_[151]); // 150
    plistsub_.push_back(plist_[150]); // 151
    plistsub_.push_back(plist_[152]); // 152
    plistsub_.push_back(plist_[153]); // 153
    plistsub_.push_back(plist_[154]); // 154
    plistsub_.push_back(plist_[155]); // 155
    plistsub_.push_back(plist_[156]); // 156
    plistsub_.push_back(plist_[157]); // 157
    plistsub_.push_back(plist_[158]); // 158
    plistsub_.push_back(plist_[159]); // 159
    plistsub_.push_back(plist_[160]); // 160
    plistsub_.push_back(plist_[161]); // 161
    plistsub_.push_back(plist_[162]); // 162
    plistsub_.push_back(plist_[164]); // 163
    plistsub_.push_back(plist_[163]); // 164
    plistsub_.push_back(plist_[165]); // 165
    plistsub_.push_back(plist_[166]); // 166
    plistsub_.push_back(plist_[167]); // 167
    plistsub_.push_back(plist_[168]); // 168
    plistsub_.push_back(plist_[169]); // 169
    plistsub_.push_back(plist_[170]); // 170
    plistsub_.push_back(plist_[171]); // 171
    ConstraintP2PAngle * c0=new ConstraintP2PAngle();
    c0->pvec.push_back(plist_[16]);
    c0->pvec.push_back(plist_[17]);
    c0->pvec.push_back(plist_[12]);
    c0->pvec.push_back(plist_[13]);
    c0->pvec.push_back(plist_[19]);
    c0->origpvec=c0->pvec;
    c0->rescale();
    clist_.push_back(c0); // addresses = 0x41f6870,0x40d90c0,0x4096550,0x41f6810,0x4191ef0
    ConstraintP2PAngle * c1=new ConstraintP2PAngle();
    c1->pvec.push_back(plist_[16]);
    c1->pvec.push_back(plist_[17]);
    c1->pvec.push_back(plist_[14]);
    c1->pvec.push_back(plist_[15]);
    c1->pvec.push_back(plist_[20]);
    c1->origpvec=c1->pvec;
    c1->rescale();
    clist_.push_back(c1); // addresses = 0x41f6870,0x40d90c0,0x41f6830,0x41f6850,0x4191f10
    ConstraintP2PDistance * c2=new ConstraintP2PDistance();
    c2->pvec.push_back(plist_[16]);
    c2->pvec.push_back(plist_[17]);
    c2->pvec.push_back(plist_[12]);
    c2->pvec.push_back(plist_[13]);
    c2->pvec.push_back(plist_[18]);
    c2->origpvec=c2->pvec;
    c2->rescale();
    clist_.push_back(c2); // addresses = 0x41f6870,0x40d90c0,0x4096550,0x41f6810,0x40d90e0
    ConstraintP2PDistance * c3=new ConstraintP2PDistance();
    c3->pvec.push_back(plist_[16]);
    c3->pvec.push_back(plist_[17]);
    c3->pvec.push_back(plist_[14]);
    c3->pvec.push_back(plist_[15]);
    c3->pvec.push_back(plist_[18]);
    c3->origpvec=c3->pvec;
    c3->rescale();
    clist_.push_back(c3); // addresses = 0x41f6870,0x40d90c0,0x41f6830,0x41f6850,0x40d90e0
    ConstraintP2PAngle * c4=new ConstraintP2PAngle();
    c4->pvec.push_back(plist_[25]);
    c4->pvec.push_back(plist_[26]);
    c4->pvec.push_back(plist_[21]);
    c4->pvec.push_back(plist_[22]);
    c4->pvec.push_back(plist_[28]);
    c4->origpvec=c4->pvec;
    c4->rescale();
    clist_.push_back(c4); // addresses = 0x41b87f0,0x41b8810,0x41c8050,0x41ee8f0,0x4178990
    ConstraintP2PAngle * c5=new ConstraintP2PAngle();
    c5->pvec.push_back(plist_[25]);
    c5->pvec.push_back(plist_[26]);
    c5->pvec.push_back(plist_[23]);
    c5->pvec.push_back(plist_[24]);
    c5->pvec.push_back(plist_[29]);
    c5->origpvec=c5->pvec;
    c5->rescale();
    clist_.push_back(c5); // addresses = 0x41b87f0,0x41b8810,0x41bfa70,0x41ee910,0x10cc6f0
    ConstraintP2PDistance * c6=new ConstraintP2PDistance();
    c6->pvec.push_back(plist_[25]);
    c6->pvec.push_back(plist_[26]);
    c6->pvec.push_back(plist_[21]);
    c6->pvec.push_back(plist_[22]);
    c6->pvec.push_back(plist_[27]);
    c6->origpvec=c6->pvec;
    c6->rescale();
    clist_.push_back(c6); // addresses = 0x41b87f0,0x41b8810,0x41c8050,0x41ee8f0,0x4178970
    ConstraintP2PDistance * c7=new ConstraintP2PDistance();
    c7->pvec.push_back(plist_[25]);
    c7->pvec.push_back(plist_[26]);
    c7->pvec.push_back(plist_[23]);
    c7->pvec.push_back(plist_[24]);
    c7->pvec.push_back(plist_[27]);
    c7->origpvec=c7->pvec;
    c7->rescale();
    clist_.push_back(c7); // addresses = 0x41b87f0,0x41b8810,0x41bfa70,0x41ee910,0x4178970
    ConstraintP2PAngle * c8=new ConstraintP2PAngle();
    c8->pvec.push_back(plist_[34]);
    c8->pvec.push_back(plist_[35]);
    c8->pvec.push_back(plist_[30]);
    c8->pvec.push_back(plist_[31]);
    c8->pvec.push_back(plist_[37]);
    c8->origpvec=c8->pvec;
    c8->rescale();
    clist_.push_back(c8); // addresses = 0x405f510,0x41592f0,0x419fce0,0x419fda0,0x417a5c0
    ConstraintP2PAngle * c9=new ConstraintP2PAngle();
    c9->pvec.push_back(plist_[34]);
    c9->pvec.push_back(plist_[35]);
    c9->pvec.push_back(plist_[32]);
    c9->pvec.push_back(plist_[33]);
    c9->pvec.push_back(plist_[38]);
    c9->origpvec=c9->pvec;
    c9->rescale();
    clist_.push_back(c9); // addresses = 0x405f510,0x41592f0,0x419fd00,0x405f4f0,0x417a5e0
    ConstraintP2PDistance * c10=new ConstraintP2PDistance();
    c10->pvec.push_back(plist_[34]);
    c10->pvec.push_back(plist_[35]);
    c10->pvec.push_back(plist_[30]);
    c10->pvec.push_back(plist_[31]);
    c10->pvec.push_back(plist_[36]);
    c10->origpvec=c10->pvec;
    c10->rescale();
    clist_.push_back(c10); // addresses = 0x405f510,0x41592f0,0x419fce0,0x419fda0,0x4159310
    ConstraintP2PDistance * c11=new ConstraintP2PDistance();
    c11->pvec.push_back(plist_[34]);
    c11->pvec.push_back(plist_[35]);
    c11->pvec.push_back(plist_[32]);
    c11->pvec.push_back(plist_[33]);
    c11->pvec.push_back(plist_[36]);
    c11->origpvec=c11->pvec;
    c11->rescale();
    clist_.push_back(c11); // addresses = 0x405f510,0x41592f0,0x419fd00,0x405f4f0,0x4159310
    ConstraintP2PAngle * c12=new ConstraintP2PAngle();
    c12->pvec.push_back(plist_[47]);
    c12->pvec.push_back(plist_[48]);
    c12->pvec.push_back(plist_[43]);
    c12->pvec.push_back(plist_[44]);
    c12->pvec.push_back(plist_[50]);
    c12->origpvec=c12->pvec;
    c12->rescale();
    clist_.push_back(c12); // addresses = 0x4186710,0x40b3790,0x42d6550,0x3d7e640,0x4209ba0
    ConstraintP2PAngle * c13=new ConstraintP2PAngle();
    c13->pvec.push_back(plist_[47]);
    c13->pvec.push_back(plist_[48]);
    c13->pvec.push_back(plist_[45]);
    c13->pvec.push_back(plist_[46]);
    c13->pvec.push_back(plist_[51]);
    c13->origpvec=c13->pvec;
    c13->rescale();
    clist_.push_back(c13); // addresses = 0x4186710,0x40b3790,0x3d6b910,0x42cb0d0,0x40b21f0
    ConstraintP2PDistance * c14=new ConstraintP2PDistance();
    c14->pvec.push_back(plist_[47]);
    c14->pvec.push_back(plist_[48]);
    c14->pvec.push_back(plist_[43]);
    c14->pvec.push_back(plist_[44]);
    c14->pvec.push_back(plist_[49]);
    c14->origpvec=c14->pvec;
    c14->rescale();
    clist_.push_back(c14); // addresses = 0x4186710,0x40b3790,0x42d6550,0x3d7e640,0x40b5190
    ConstraintP2PDistance * c15=new ConstraintP2PDistance();
    c15->pvec.push_back(plist_[47]);
    c15->pvec.push_back(plist_[48]);
    c15->pvec.push_back(plist_[45]);
    c15->pvec.push_back(plist_[46]);
    c15->pvec.push_back(plist_[49]);
    c15->origpvec=c15->pvec;
    c15->rescale();
    clist_.push_back(c15); // addresses = 0x4186710,0x40b3790,0x3d6b910,0x42cb0d0,0x40b5190
    ConstraintP2PAngle * c16=new ConstraintP2PAngle();
    c16->pvec.push_back(plist_[56]);
    c16->pvec.push_back(plist_[57]);
    c16->pvec.push_back(plist_[52]);
    c16->pvec.push_back(plist_[53]);
    c16->pvec.push_back(plist_[59]);
    c16->origpvec=c16->pvec;
    c16->rescale();
    clist_.push_back(c16); // addresses = 0x42caaa0,0x42caac0,0x41fd260,0x41f69f0,0x42cab00
    ConstraintP2PAngle * c17=new ConstraintP2PAngle();
    c17->pvec.push_back(plist_[56]);
    c17->pvec.push_back(plist_[57]);
    c17->pvec.push_back(plist_[54]);
    c17->pvec.push_back(plist_[55]);
    c17->pvec.push_back(plist_[60]);
    c17->origpvec=c17->pvec;
    c17->rescale();
    clist_.push_back(c17); // addresses = 0x42caaa0,0x42caac0,0x41fd280,0x42caa80,0x42cab20
    ConstraintP2PDistance * c18=new ConstraintP2PDistance();
    c18->pvec.push_back(plist_[56]);
    c18->pvec.push_back(plist_[57]);
    c18->pvec.push_back(plist_[52]);
    c18->pvec.push_back(plist_[53]);
    c18->pvec.push_back(plist_[58]);
    c18->origpvec=c18->pvec;
    c18->rescale();
    clist_.push_back(c18); // addresses = 0x42caaa0,0x42caac0,0x41fd260,0x41f69f0,0x42caae0
    ConstraintP2PDistance * c19=new ConstraintP2PDistance();
    c19->pvec.push_back(plist_[56]);
    c19->pvec.push_back(plist_[57]);
    c19->pvec.push_back(plist_[54]);
    c19->pvec.push_back(plist_[55]);
    c19->pvec.push_back(plist_[58]);
    c19->origpvec=c19->pvec;
    c19->rescale();
    clist_.push_back(c19); // addresses = 0x42caaa0,0x42caac0,0x41fd280,0x42caa80,0x42caae0
    ConstraintP2PAngle * c20=new ConstraintP2PAngle();
    c20->pvec.push_back(plist_[65]);
    c20->pvec.push_back(plist_[66]);
    c20->pvec.push_back(plist_[61]);
    c20->pvec.push_back(plist_[62]);
    c20->pvec.push_back(plist_[68]);
    c20->origpvec=c20->pvec;
    c20->rescale();
    clist_.push_back(c20); // addresses = 0x41fea60,0x42d4a10,0x42db6c0,0x42db6a0,0x41f6a10
    ConstraintP2PAngle * c21=new ConstraintP2PAngle();
    c21->pvec.push_back(plist_[65]);
    c21->pvec.push_back(plist_[66]);
    c21->pvec.push_back(plist_[63]);
    c21->pvec.push_back(plist_[64]);
    c21->pvec.push_back(plist_[69]);
    c21->origpvec=c21->pvec;
    c21->rescale();
    clist_.push_back(c21); // addresses = 0x41fea60,0x42d4a10,0x41febe0,0x41fec00,0x41f6a30
    ConstraintP2PDistance * c22=new ConstraintP2PDistance();
    c22->pvec.push_back(plist_[65]);
    c22->pvec.push_back(plist_[66]);
    c22->pvec.push_back(plist_[61]);
    c22->pvec.push_back(plist_[62]);
    c22->pvec.push_back(plist_[67]);
    c22->origpvec=c22->pvec;
    c22->rescale();
    clist_.push_back(c22); // addresses = 0x41fea60,0x42d4a10,0x42db6c0,0x42db6a0,0x42d4a30
    ConstraintP2PDistance * c23=new ConstraintP2PDistance();
    c23->pvec.push_back(plist_[65]);
    c23->pvec.push_back(plist_[66]);
    c23->pvec.push_back(plist_[63]);
    c23->pvec.push_back(plist_[64]);
    c23->pvec.push_back(plist_[67]);
    c23->origpvec=c23->pvec;
    c23->rescale();
    clist_.push_back(c23); // addresses = 0x41fea60,0x42d4a10,0x41febe0,0x41fec00,0x42d4a30
    ConstraintP2PAngle * c24=new ConstraintP2PAngle();
    c24->pvec.push_back(plist_[74]);
    c24->pvec.push_back(plist_[75]);
    c24->pvec.push_back(plist_[70]);
    c24->pvec.push_back(plist_[71]);
    c24->pvec.push_back(plist_[77]);
    c24->origpvec=c24->pvec;
    c24->rescale();
    clist_.push_back(c24); // addresses = 0x41f4930,0x41f4950,0x42dc090,0x41f48d0,0x41f4990
    ConstraintP2PAngle * c25=new ConstraintP2PAngle();
    c25->pvec.push_back(plist_[74]);
    c25->pvec.push_back(plist_[75]);
    c25->pvec.push_back(plist_[72]);
    c25->pvec.push_back(plist_[73]);
    c25->pvec.push_back(plist_[78]);
    c25->origpvec=c25->pvec;
    c25->rescale();
    clist_.push_back(c25); // addresses = 0x41f4930,0x41f4950,0x41f48f0,0x41f4910,0x41f49b0
    ConstraintP2PDistance * c26=new ConstraintP2PDistance();
    c26->pvec.push_back(plist_[74]);
    c26->pvec.push_back(plist_[75]);
    c26->pvec.push_back(plist_[70]);
    c26->pvec.push_back(plist_[71]);
    c26->pvec.push_back(plist_[76]);
    c26->origpvec=c26->pvec;
    c26->rescale();
    clist_.push_back(c26); // addresses = 0x41f4930,0x41f4950,0x42dc090,0x41f48d0,0x41f4970
    ConstraintP2PDistance * c27=new ConstraintP2PDistance();
    c27->pvec.push_back(plist_[74]);
    c27->pvec.push_back(plist_[75]);
    c27->pvec.push_back(plist_[72]);
    c27->pvec.push_back(plist_[73]);
    c27->pvec.push_back(plist_[76]);
    c27->origpvec=c27->pvec;
    c27->rescale();
    clist_.push_back(c27); // addresses = 0x41f4930,0x41f4950,0x41f48f0,0x41f4910,0x41f4970
    ConstraintP2PAngle * c28=new ConstraintP2PAngle();
    c28->pvec.push_back(plist_[83]);
    c28->pvec.push_back(plist_[84]);
    c28->pvec.push_back(plist_[79]);
    c28->pvec.push_back(plist_[80]);
    c28->pvec.push_back(plist_[86]);
    c28->origpvec=c28->pvec;
    c28->rescale();
    clist_.push_back(c28); // addresses = 0x42db110,0x40b5b20,0x42dacf0,0x42dacd0,0x41fe2b0
    ConstraintP2PAngle * c29=new ConstraintP2PAngle();
    c29->pvec.push_back(plist_[83]);
    c29->pvec.push_back(plist_[84]);
    c29->pvec.push_back(plist_[81]);
    c29->pvec.push_back(plist_[82]);
    c29->pvec.push_back(plist_[87]);
    c29->origpvec=c29->pvec;
    c29->rescale();
    clist_.push_back(c29); // addresses = 0x42db110,0x40b5b20,0x42db0d0,0x42db0f0,0x41fe2d0
    ConstraintP2PDistance * c30=new ConstraintP2PDistance();
    c30->pvec.push_back(plist_[83]);
    c30->pvec.push_back(plist_[84]);
    c30->pvec.push_back(plist_[79]);
    c30->pvec.push_back(plist_[80]);
    c30->pvec.push_back(plist_[85]);
    c30->origpvec=c30->pvec;
    c30->rescale();
    clist_.push_back(c30); // addresses = 0x42db110,0x40b5b20,0x42dacf0,0x42dacd0,0x40b5b40
    ConstraintP2PDistance * c31=new ConstraintP2PDistance();
    c31->pvec.push_back(plist_[83]);
    c31->pvec.push_back(plist_[84]);
    c31->pvec.push_back(plist_[81]);
    c31->pvec.push_back(plist_[82]);
    c31->pvec.push_back(plist_[85]);
    c31->origpvec=c31->pvec;
    c31->rescale();
    clist_.push_back(c31); // addresses = 0x42db110,0x40b5b20,0x42db0d0,0x42db0f0,0x40b5b40
    ConstraintP2PAngle * c32=new ConstraintP2PAngle();
    c32->pvec.push_back(plist_[101]);
    c32->pvec.push_back(plist_[102]);
    c32->pvec.push_back(plist_[97]);
    c32->pvec.push_back(plist_[98]);
    c32->pvec.push_back(plist_[104]);
    c32->origpvec=c32->pvec;
    c32->rescale();
    clist_.push_back(c32); // addresses = 0x41fc0e0,0x41865e0,0x41fc060,0x41fc080,0x40b67a0
    ConstraintP2PAngle * c33=new ConstraintP2PAngle();
    c33->pvec.push_back(plist_[101]);
    c33->pvec.push_back(plist_[102]);
    c33->pvec.push_back(plist_[99]);
    c33->pvec.push_back(plist_[100]);
    c33->pvec.push_back(plist_[105]);
    c33->origpvec=c33->pvec;
    c33->rescale();
    clist_.push_back(c33); // addresses = 0x41fc0e0,0x41865e0,0x41fc0a0,0x41fc0c0,0x40b67c0
    ConstraintP2PDistance * c34=new ConstraintP2PDistance();
    c34->pvec.push_back(plist_[101]);
    c34->pvec.push_back(plist_[102]);
    c34->pvec.push_back(plist_[97]);
    c34->pvec.push_back(plist_[98]);
    c34->pvec.push_back(plist_[103]);
    c34->origpvec=c34->pvec;
    c34->rescale();
    clist_.push_back(c34); // addresses = 0x41fc0e0,0x41865e0,0x41fc060,0x41fc080,0x4186600
    ConstraintP2PDistance * c35=new ConstraintP2PDistance();
    c35->pvec.push_back(plist_[101]);
    c35->pvec.push_back(plist_[102]);
    c35->pvec.push_back(plist_[99]);
    c35->pvec.push_back(plist_[100]);
    c35->pvec.push_back(plist_[103]);
    c35->origpvec=c35->pvec;
    c35->rescale();
    clist_.push_back(c35); // addresses = 0x41fc0e0,0x41865e0,0x41fc0a0,0x41fc0c0,0x4186600
    ConstraintP2PAngle * c36=new ConstraintP2PAngle();
    c36->pvec.push_back(plist_[110]);
    c36->pvec.push_back(plist_[111]);
    c36->pvec.push_back(plist_[106]);
    c36->pvec.push_back(plist_[107]);
    c36->pvec.push_back(plist_[113]);
    c36->origpvec=c36->pvec;
    c36->rescale();
    clist_.push_back(c36); // addresses = 0x42d8db0,0x42d8dd0,0x42d8910,0x42d88f0,0x42d8e10
    ConstraintP2PAngle * c37=new ConstraintP2PAngle();
    c37->pvec.push_back(plist_[110]);
    c37->pvec.push_back(plist_[111]);
    c37->pvec.push_back(plist_[108]);
    c37->pvec.push_back(plist_[109]);
    c37->pvec.push_back(plist_[114]);
    c37->origpvec=c37->pvec;
    c37->rescale();
    clist_.push_back(c37); // addresses = 0x42d8db0,0x42d8dd0,0x42d8d70,0x42d8d90,0x42d8e30
    ConstraintP2PDistance * c38=new ConstraintP2PDistance();
    c38->pvec.push_back(plist_[110]);
    c38->pvec.push_back(plist_[111]);
    c38->pvec.push_back(plist_[106]);
    c38->pvec.push_back(plist_[107]);
    c38->pvec.push_back(plist_[112]);
    c38->origpvec=c38->pvec;
    c38->rescale();
    clist_.push_back(c38); // addresses = 0x42d8db0,0x42d8dd0,0x42d8910,0x42d88f0,0x42d8df0
    ConstraintP2PDistance * c39=new ConstraintP2PDistance();
    c39->pvec.push_back(plist_[110]);
    c39->pvec.push_back(plist_[111]);
    c39->pvec.push_back(plist_[108]);
    c39->pvec.push_back(plist_[109]);
    c39->pvec.push_back(plist_[112]);
    c39->origpvec=c39->pvec;
    c39->rescale();
    clist_.push_back(c39); // addresses = 0x42d8db0,0x42d8dd0,0x42d8d70,0x42d8d90,0x42d8df0
    ConstraintP2PAngle * c40=new ConstraintP2PAngle();
    c40->pvec.push_back(plist_[123]);
    c40->pvec.push_back(plist_[124]);
    c40->pvec.push_back(plist_[119]);
    c40->pvec.push_back(plist_[120]);
    c40->pvec.push_back(plist_[126]);
    c40->origpvec=c40->pvec;
    c40->rescale();
    clist_.push_back(c40); // addresses = 0x42ea1d0,0x42ea1f0,0x42ea150,0x42ea170,0x42ea230
    ConstraintP2PAngle * c41=new ConstraintP2PAngle();
    c41->pvec.push_back(plist_[123]);
    c41->pvec.push_back(plist_[124]);
    c41->pvec.push_back(plist_[121]);
    c41->pvec.push_back(plist_[122]);
    c41->pvec.push_back(plist_[127]);
    c41->origpvec=c41->pvec;
    c41->rescale();
    clist_.push_back(c41); // addresses = 0x42ea1d0,0x42ea1f0,0x42ea190,0x42ea1b0,0x42ea250
    ConstraintP2PDistance * c42=new ConstraintP2PDistance();
    c42->pvec.push_back(plist_[123]);
    c42->pvec.push_back(plist_[124]);
    c42->pvec.push_back(plist_[119]);
    c42->pvec.push_back(plist_[120]);
    c42->pvec.push_back(plist_[125]);
    c42->origpvec=c42->pvec;
    c42->rescale();
    clist_.push_back(c42); // addresses = 0x42ea1d0,0x42ea1f0,0x42ea150,0x42ea170,0x42ea210
    ConstraintP2PDistance * c43=new ConstraintP2PDistance();
    c43->pvec.push_back(plist_[123]);
    c43->pvec.push_back(plist_[124]);
    c43->pvec.push_back(plist_[121]);
    c43->pvec.push_back(plist_[122]);
    c43->pvec.push_back(plist_[125]);
    c43->origpvec=c43->pvec;
    c43->rescale();
    clist_.push_back(c43); // addresses = 0x42ea1d0,0x42ea1f0,0x42ea190,0x42ea1b0,0x42ea210
    ConstraintP2PAngle * c44=new ConstraintP2PAngle();
    c44->pvec.push_back(plist_[136]);
    c44->pvec.push_back(plist_[137]);
    c44->pvec.push_back(plist_[132]);
    c44->pvec.push_back(plist_[133]);
    c44->pvec.push_back(plist_[139]);
    c44->origpvec=c44->pvec;
    c44->rescale();
    clist_.push_back(c44); // addresses = 0x4200e70,0x4200e90,0x4200df0,0x4200e10,0x4200ed0
    ConstraintP2PAngle * c45=new ConstraintP2PAngle();
    c45->pvec.push_back(plist_[136]);
    c45->pvec.push_back(plist_[137]);
    c45->pvec.push_back(plist_[134]);
    c45->pvec.push_back(plist_[135]);
    c45->pvec.push_back(plist_[140]);
    c45->origpvec=c45->pvec;
    c45->rescale();
    clist_.push_back(c45); // addresses = 0x4200e70,0x4200e90,0x4200e30,0x4200e50,0x4200ef0
    ConstraintP2PDistance * c46=new ConstraintP2PDistance();
    c46->pvec.push_back(plist_[136]);
    c46->pvec.push_back(plist_[137]);
    c46->pvec.push_back(plist_[132]);
    c46->pvec.push_back(plist_[133]);
    c46->pvec.push_back(plist_[138]);
    c46->origpvec=c46->pvec;
    c46->rescale();
    clist_.push_back(c46); // addresses = 0x4200e70,0x4200e90,0x4200df0,0x4200e10,0x4200eb0
    ConstraintP2PDistance * c47=new ConstraintP2PDistance();
    c47->pvec.push_back(plist_[136]);
    c47->pvec.push_back(plist_[137]);
    c47->pvec.push_back(plist_[134]);
    c47->pvec.push_back(plist_[135]);
    c47->pvec.push_back(plist_[138]);
    c47->origpvec=c47->pvec;
    c47->rescale();
    clist_.push_back(c47); // addresses = 0x4200e70,0x4200e90,0x4200e30,0x4200e50,0x4200eb0
    ConstraintP2PAngle * c48=new ConstraintP2PAngle();
    c48->pvec.push_back(plist_[145]);
    c48->pvec.push_back(plist_[146]);
    c48->pvec.push_back(plist_[141]);
    c48->pvec.push_back(plist_[142]);
    c48->pvec.push_back(plist_[148]);
    c48->origpvec=c48->pvec;
    c48->rescale();
    clist_.push_back(c48); // addresses = 0x42ebe40,0x42ebe60,0x42eba20,0x42eba00,0x42ebea0
    ConstraintP2PAngle * c49=new ConstraintP2PAngle();
    c49->pvec.push_back(plist_[145]);
    c49->pvec.push_back(plist_[146]);
    c49->pvec.push_back(plist_[143]);
    c49->pvec.push_back(plist_[144]);
    c49->pvec.push_back(plist_[149]);
    c49->origpvec=c49->pvec;
    c49->rescale();
    clist_.push_back(c49); // addresses = 0x42ebe40,0x42ebe60,0x42ebe00,0x42ebe20,0x42ebec0
    ConstraintP2PDistance * c50=new ConstraintP2PDistance();
    c50->pvec.push_back(plist_[145]);
    c50->pvec.push_back(plist_[146]);
    c50->pvec.push_back(plist_[141]);
    c50->pvec.push_back(plist_[142]);
    c50->pvec.push_back(plist_[147]);
    c50->origpvec=c50->pvec;
    c50->rescale();
    clist_.push_back(c50); // addresses = 0x42ebe40,0x42ebe60,0x42eba20,0x42eba00,0x42ebe80
    ConstraintP2PDistance * c51=new ConstraintP2PDistance();
    c51->pvec.push_back(plist_[145]);
    c51->pvec.push_back(plist_[146]);
    c51->pvec.push_back(plist_[143]);
    c51->pvec.push_back(plist_[144]);
    c51->pvec.push_back(plist_[147]);
    c51->origpvec=c51->pvec;
    c51->rescale();
    clist_.push_back(c51); // addresses = 0x42ebe40,0x42ebe60,0x42ebe00,0x42ebe20,0x42ebe80
    ConstraintP2PAngle * c52=new ConstraintP2PAngle();
    c52->pvec.push_back(plist_[158]);
    c52->pvec.push_back(plist_[159]);
    c52->pvec.push_back(plist_[154]);
    c52->pvec.push_back(plist_[155]);
    c52->pvec.push_back(plist_[161]);
    c52->origpvec=c52->pvec;
    c52->rescale();
    clist_.push_back(c52); // addresses = 0x42ecb90,0x42ecbb0,0x42ecb10,0x42ecb30,0x42ecbf0
    ConstraintP2PAngle * c53=new ConstraintP2PAngle();
    c53->pvec.push_back(plist_[158]);
    c53->pvec.push_back(plist_[159]);
    c53->pvec.push_back(plist_[156]);
    c53->pvec.push_back(plist_[157]);
    c53->pvec.push_back(plist_[162]);
    c53->origpvec=c53->pvec;
    c53->rescale();
    clist_.push_back(c53); // addresses = 0x42ecb90,0x42ecbb0,0x42ecb50,0x42ecb70,0x42ecc10
    ConstraintP2PDistance * c54=new ConstraintP2PDistance();
    c54->pvec.push_back(plist_[158]);
    c54->pvec.push_back(plist_[159]);
    c54->pvec.push_back(plist_[154]);
    c54->pvec.push_back(plist_[155]);
    c54->pvec.push_back(plist_[160]);
    c54->origpvec=c54->pvec;
    c54->rescale();
    clist_.push_back(c54); // addresses = 0x42ecb90,0x42ecbb0,0x42ecb10,0x42ecb30,0x42ecbd0
    ConstraintP2PDistance * c55=new ConstraintP2PDistance();
    c55->pvec.push_back(plist_[158]);
    c55->pvec.push_back(plist_[159]);
    c55->pvec.push_back(plist_[156]);
    c55->pvec.push_back(plist_[157]);
    c55->pvec.push_back(plist_[160]);
    c55->origpvec=c55->pvec;
    c55->rescale();
    clist_.push_back(c55); // addresses = 0x42ecb90,0x42ecbb0,0x42ecb50,0x42ecb70,0x42ecbd0
    ConstraintP2PAngle * c56=new ConstraintP2PAngle();
    c56->pvec.push_back(plist_[167]);
    c56->pvec.push_back(plist_[168]);
    c56->pvec.push_back(plist_[163]);
    c56->pvec.push_back(plist_[164]);
    c56->pvec.push_back(plist_[170]);
    c56->origpvec=c56->pvec;
    c56->rescale();
    clist_.push_back(c56); // addresses = 0x42ed720,0x42ed740,0x42ed300,0x42ed2e0,0x42ed780
    ConstraintP2PAngle * c57=new ConstraintP2PAngle();
    c57->pvec.push_back(plist_[167]);
    c57->pvec.push_back(plist_[168]);
    c57->pvec.push_back(plist_[165]);
    c57->pvec.push_back(plist_[166]);
    c57->pvec.push_back(plist_[171]);
    c57->origpvec=c57->pvec;
    c57->rescale();
    clist_.push_back(c57); // addresses = 0x42ed720,0x42ed740,0x42ed6e0,0x42ed700,0x42ed7a0
    ConstraintP2PDistance * c58=new ConstraintP2PDistance();
    c58->pvec.push_back(plist_[167]);
    c58->pvec.push_back(plist_[168]);
    c58->pvec.push_back(plist_[163]);
    c58->pvec.push_back(plist_[164]);
    c58->pvec.push_back(plist_[169]);
    c58->origpvec=c58->pvec;
    c58->rescale();
    clist_.push_back(c58); // addresses = 0x42ed720,0x42ed740,0x42ed300,0x42ed2e0,0x42ed760
    ConstraintP2PDistance * c59=new ConstraintP2PDistance();
    c59->pvec.push_back(plist_[167]);
    c59->pvec.push_back(plist_[168]);
    c59->pvec.push_back(plist_[165]);
    c59->pvec.push_back(plist_[166]);
    c59->pvec.push_back(plist_[169]);
    c59->origpvec=c59->pvec;
    c59->rescale();
    clist_.push_back(c59); // addresses = 0x42ed720,0x42ed740,0x42ed6e0,0x42ed700,0x42ed760
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 0 address: 0x3d97740
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 1 address: 0x4200ce0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(1)); // 2 address: 0x420cb50
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 3 address: 0x420cb70
    ConstraintPointOnLine * c60=new ConstraintPointOnLine();
    c60->pvec.push_back(plist_[0]);
    c60->pvec.push_back(plist_[1]);
    c60->pvec.push_back(clist_params_[0]);
    c60->pvec.push_back(clist_params_[1]);
    c60->pvec.push_back(clist_params_[2]);
    c60->pvec.push_back(clist_params_[3]);
    c60->origpvec=c60->pvec;
    c60->rescale();
    clist_.push_back(c60); // addresses = 0x4208ea0,0x42dce00,0x3d97740,0x4200ce0,0x420cb50,0x420cb70
    ConstraintPointOnLine * c61=new ConstraintPointOnLine();
    c61->pvec.push_back(plist_[3]);
    c61->pvec.push_back(plist_[4]);
    c61->pvec.push_back(clist_params_[0]);
    c61->pvec.push_back(clist_params_[1]);
    c61->pvec.push_back(clist_params_[2]);
    c61->pvec.push_back(clist_params_[3]);
    c61->origpvec=c61->pvec;
    c61->rescale();
    clist_.push_back(c61); // addresses = 0x4161500,0x41e7780,0x3d97740,0x4200ce0,0x420cb50,0x420cb70
    ConstraintPointOnLine * c62=new ConstraintPointOnLine();
    c62->pvec.push_back(plist_[9]);
    c62->pvec.push_back(plist_[10]);
    c62->pvec.push_back(clist_params_[0]);
    c62->pvec.push_back(clist_params_[1]);
    c62->pvec.push_back(clist_params_[2]);
    c62->pvec.push_back(clist_params_[3]);
    c62->origpvec=c62->pvec;
    c62->rescale();
    clist_.push_back(c62); // addresses = 0x42dc760,0x41f0570,0x3d97740,0x4200ce0,0x420cb50,0x420cb70
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 4 address: 0x42edee0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 5 address: 0x42ee2e0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(0)); // 6 address: 0x42edec0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(1)); // 7 address: 0x42ee300
    ConstraintMidpointOnLine * c63=new ConstraintMidpointOnLine();
    c63->pvec.push_back(plist_[9]);
    c63->pvec.push_back(plist_[10]);
    c63->pvec.push_back(plist_[0]);
    c63->pvec.push_back(plist_[1]);
    c63->pvec.push_back(clist_params_[4]);
    c63->pvec.push_back(clist_params_[5]);
    c63->pvec.push_back(clist_params_[6]);
    c63->pvec.push_back(clist_params_[7]);
    c63->origpvec=c63->pvec;
    c63->rescale();
    clist_.push_back(c63); // addresses = 0x42dc760,0x41f0570,0x4208ea0,0x42dce00,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(200)); // 8 address: 0x42eea00
    ConstraintP2PDistance * c64=new ConstraintP2PDistance();
    c64->pvec.push_back(plist_[0]);
    c64->pvec.push_back(plist_[1]);
    c64->pvec.push_back(plist_[9]);
    c64->pvec.push_back(plist_[10]);
    c64->pvec.push_back(clist_params_[8]);
    c64->origpvec=c64->pvec;
    c64->rescale();
    clist_.push_back(c64); // addresses = 0x4208ea0,0x42dce00,0x42dc760,0x41f0570,0x42eea00
    ConstraintPerpendicular * c65=new ConstraintPerpendicular();
    c65->pvec.push_back(plist_[3]);
    c65->pvec.push_back(plist_[4]);
    c65->pvec.push_back(plist_[6]);
    c65->pvec.push_back(plist_[7]);
    c65->pvec.push_back(clist_params_[4]);
    c65->pvec.push_back(clist_params_[5]);
    c65->pvec.push_back(clist_params_[6]);
    c65->pvec.push_back(clist_params_[7]);
    c65->origpvec=c65->pvec;
    c65->rescale();
    clist_.push_back(c65); // addresses = 0x4161500,0x41e7780,0x41efe80,0x4202700,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    ConstraintMidpointOnLine * c66=new ConstraintMidpointOnLine();
    c66->pvec.push_back(plist_[3]);
    c66->pvec.push_back(plist_[4]);
    c66->pvec.push_back(plist_[6]);
    c66->pvec.push_back(plist_[7]);
    c66->pvec.push_back(clist_params_[4]);
    c66->pvec.push_back(clist_params_[5]);
    c66->pvec.push_back(clist_params_[6]);
    c66->pvec.push_back(clist_params_[7]);
    c66->origpvec=c66->pvec;
    c66->rescale();
    clist_.push_back(c66); // addresses = 0x4161500,0x41e7780,0x41efe80,0x4202700,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(75)); // 9 address: 0x42ee8c0
    ConstraintP2PDistance * c67=new ConstraintP2PDistance();
    c67->pvec.push_back(plist_[3]);
    c67->pvec.push_back(plist_[4]);
    c67->pvec.push_back(plist_[6]);
    c67->pvec.push_back(plist_[7]);
    c67->pvec.push_back(clist_params_[9]);
    c67->origpvec=c67->pvec;
    c67->rescale();
    clist_.push_back(c67); // addresses = 0x4161500,0x41e7780,0x41efe80,0x4202700,0x42ee8c0
    clist_.push_back(new ConstraintEqual(plist_[11],plist_[8])); // addresses = 0x41f0830,0x42c9940
    clist_.push_back(new ConstraintEqual(plist_[8],plist_[5])); // addresses = 0x42c9940,0x4205700
    clist_.push_back(new ConstraintEqual(plist_[5],plist_[2])); // addresses = 0x4205700,0x42bd340
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(10)); // 10 address: 0x42ef700
    clist_.push_back(new ConstraintEqual(plist_[11],clist_params_[10])); // addresses = 0x41f0830,0x42ef700
    clist_.push_back(new ConstraintEqual(plist_[16],plist_[0])); // addresses = 0x41f6870,0x4208ea0
    clist_.push_back(new ConstraintEqual(plist_[17],plist_[1])); // addresses = 0x40d90c0,0x42dce00
    clist_.push_back(new ConstraintEqual(plist_[21],plist_[12])); // addresses = 0x41c8050,0x4096550
    clist_.push_back(new ConstraintEqual(plist_[22],plist_[13])); // addresses = 0x41ee8f0,0x41f6810
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(20)); // 11 address: 0x42f0930
    clist_.push_back(new ConstraintEqual(plist_[18],clist_params_[11])); // addresses = 0x40d90e0,0x42f0930
    ConstraintTangentCircumf * c77=new ConstraintTangentCircumf(false);
    c77->pvec.push_back(plist_[25]);
    c77->pvec.push_back(plist_[26]);
    c77->pvec.push_back(plist_[16]);
    c77->pvec.push_back(plist_[17]);
    c77->pvec.push_back(plist_[27]);
    c77->pvec.push_back(plist_[18]);
    c77->origpvec=c77->pvec;
    c77->rescale();
    clist_.push_back(c77); // addresses = 0x41b87f0,0x41b8810,0x41f6870,0x40d90c0,0x4178970,0x40d90e0
    clist_.push_back(new ConstraintEqual(plist_[32],plist_[23])); // addresses = 0x419fd00,0x41bfa70
    clist_.push_back(new ConstraintEqual(plist_[33],plist_[24])); // addresses = 0x405f4f0,0x41ee910
    ConstraintTangentCircumf * c80=new ConstraintTangentCircumf(false);
    c80->pvec.push_back(plist_[34]);
    c80->pvec.push_back(plist_[35]);
    c80->pvec.push_back(plist_[25]);
    c80->pvec.push_back(plist_[26]);
    c80->pvec.push_back(plist_[36]);
    c80->pvec.push_back(plist_[27]);
    c80->origpvec=c80->pvec;
    c80->rescale();
    clist_.push_back(c80); // addresses = 0x405f510,0x41592f0,0x41b87f0,0x41b8810,0x4159310,0x4178970
    clist_.push_back(new ConstraintEqual(plist_[34],plist_[3])); // addresses = 0x405f510,0x4161500
    clist_.push_back(new ConstraintEqual(plist_[35],plist_[4])); // addresses = 0x41592f0,0x41e7780
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(20)); // 12 address: 0x42f0280
    clist_.push_back(new ConstraintEqual(plist_[36],clist_params_[12])); // addresses = 0x4159310,0x42f0280
    ConstraintPointOnLine * c84=new ConstraintPointOnLine();
    c84->pvec.push_back(plist_[39]);
    c84->pvec.push_back(plist_[40]);
    c84->pvec.push_back(clist_params_[0]);
    c84->pvec.push_back(clist_params_[1]);
    c84->pvec.push_back(clist_params_[2]);
    c84->pvec.push_back(clist_params_[3]);
    c84->origpvec=c84->pvec;
    c84->rescale();
    clist_.push_back(c84); // addresses = 0x4151bf0,0x414c1f0,0x3d97740,0x4200ce0,0x420cb50,0x420cb70
    ConstraintPointOnLine * c85=new ConstraintPointOnLine();
    c85->pvec.push_back(plist_[41]);
    c85->pvec.push_back(plist_[42]);
    c85->pvec.push_back(clist_params_[0]);
    c85->pvec.push_back(clist_params_[1]);
    c85->pvec.push_back(clist_params_[2]);
    c85->pvec.push_back(clist_params_[3]);
    c85->origpvec=c85->pvec;
    c85->rescale();
    clist_.push_back(c85); // addresses = 0x4162f90,0x41e1170,0x3d97740,0x4200ce0,0x420cb50,0x420cb70
    clist_.push_back(new ConstraintEqual(plist_[43],plist_[30])); // addresses = 0x42d6550,0x419fce0
    clist_.push_back(new ConstraintEqual(plist_[44],plist_[31])); // addresses = 0x3d7e640,0x419fda0
    clist_.push_back(new ConstraintEqual(plist_[45],plist_[39])); // addresses = 0x3d6b910,0x4151bf0
    clist_.push_back(new ConstraintEqual(plist_[46],plist_[40])); // addresses = 0x42cb0d0,0x414c1f0
    ConstraintTangentCircumf * c90=new ConstraintTangentCircumf(false);
    c90->pvec.push_back(plist_[47]);
    c90->pvec.push_back(plist_[48]);
    c90->pvec.push_back(plist_[34]);
    c90->pvec.push_back(plist_[35]);
    c90->pvec.push_back(plist_[49]);
    c90->pvec.push_back(plist_[36]);
    c90->origpvec=c90->pvec;
    c90->rescale();
    clist_.push_back(c90); // addresses = 0x4186710,0x40b3790,0x405f510,0x41592f0,0x40b5190,0x4159310
    ConstraintP2LDistance * c91=new ConstraintP2LDistance();
    c91->pvec.push_back(plist_[47]);
    c91->pvec.push_back(plist_[48]);
    c91->pvec.push_back(plist_[39]);
    c91->pvec.push_back(plist_[40]);
    c91->pvec.push_back(plist_[41]);
    c91->pvec.push_back(plist_[42]);
    c91->pvec.push_back(plist_[49]);
    c91->origpvec=c91->pvec;
    c91->rescale();
    clist_.push_back(c91); // addresses = 0x4186710,0x40b3790,0x4151bf0,0x414c1f0,0x4162f90,0x41e1170,0x40b5190
    clist_.push_back(new ConstraintEqual(plist_[52],plist_[41])); // addresses = 0x41fd260,0x4162f90
    clist_.push_back(new ConstraintEqual(plist_[53],plist_[42])); // addresses = 0x41f69f0,0x41e1170
    clist_.push_back(new ConstraintEqual(plist_[65],plist_[6])); // addresses = 0x41fea60,0x41efe80
    clist_.push_back(new ConstraintEqual(plist_[66],plist_[7])); // addresses = 0x42d4a10,0x4202700
    clist_.push_back(new ConstraintEqual(plist_[63],plist_[54])); // addresses = 0x41febe0,0x41fd280
    clist_.push_back(new ConstraintEqual(plist_[64],plist_[55])); // addresses = 0x41fec00,0x42caa80
    ConstraintP2LDistance * c98=new ConstraintP2LDistance();
    c98->pvec.push_back(plist_[56]);
    c98->pvec.push_back(plist_[57]);
    c98->pvec.push_back(plist_[39]);
    c98->pvec.push_back(plist_[40]);
    c98->pvec.push_back(plist_[41]);
    c98->pvec.push_back(plist_[42]);
    c98->pvec.push_back(plist_[58]);
    c98->origpvec=c98->pvec;
    c98->rescale();
    clist_.push_back(c98); // addresses = 0x42caaa0,0x42caac0,0x4151bf0,0x414c1f0,0x4162f90,0x41e1170,0x42caae0
    ConstraintTangentCircumf * c99=new ConstraintTangentCircumf(false);
    c99->pvec.push_back(plist_[65]);
    c99->pvec.push_back(plist_[66]);
    c99->pvec.push_back(plist_[56]);
    c99->pvec.push_back(plist_[57]);
    c99->pvec.push_back(plist_[67]);
    c99->pvec.push_back(plist_[58]);
    c99->origpvec=c99->pvec;
    c99->rescale();
    clist_.push_back(c99); // addresses = 0x41fea60,0x42d4a10,0x42caaa0,0x42caac0,0x42d4a30,0x42caae0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(20)); // 13 address: 0x42cac50
    clist_.push_back(new ConstraintEqual(plist_[67],clist_params_[13])); // addresses = 0x42d4a30,0x42cac50
    clist_.push_back(new ConstraintEqual(plist_[70],plist_[61])); // addresses = 0x42dc090,0x42db6c0
    clist_.push_back(new ConstraintEqual(plist_[71],plist_[62])); // addresses = 0x41f48d0,0x42db6a0
    clist_.push_back(new ConstraintEqual(plist_[83],plist_[9])); // addresses = 0x42db110,0x42dc760
    clist_.push_back(new ConstraintEqual(plist_[84],plist_[10])); // addresses = 0x40b5b20,0x41f0570
    clist_.push_back(new ConstraintEqual(plist_[81],plist_[72])); // addresses = 0x42db0d0,0x41f48f0
    clist_.push_back(new ConstraintEqual(plist_[82],plist_[73])); // addresses = 0x42db0f0,0x41f4910
    ConstraintTangentCircumf * c107=new ConstraintTangentCircumf(false);
    c107->pvec.push_back(plist_[74]);
    c107->pvec.push_back(plist_[75]);
    c107->pvec.push_back(plist_[65]);
    c107->pvec.push_back(plist_[66]);
    c107->pvec.push_back(plist_[76]);
    c107->pvec.push_back(plist_[67]);
    c107->origpvec=c107->pvec;
    c107->rescale();
    clist_.push_back(c107); // addresses = 0x41f4930,0x41f4950,0x41fea60,0x42d4a10,0x41f4970,0x42d4a30
    ConstraintTangentCircumf * c108=new ConstraintTangentCircumf(false);
    c108->pvec.push_back(plist_[74]);
    c108->pvec.push_back(plist_[75]);
    c108->pvec.push_back(plist_[83]);
    c108->pvec.push_back(plist_[84]);
    c108->pvec.push_back(plist_[76]);
    c108->pvec.push_back(plist_[85]);
    c108->origpvec=c108->pvec;
    c108->rescale();
    clist_.push_back(c108); // addresses = 0x41f4930,0x41f4950,0x42db110,0x40b5b20,0x41f4970,0x40b5b40
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(20)); // 14 address: 0x42d6d30
    clist_.push_back(new ConstraintEqual(plist_[85],clist_params_[14])); // addresses = 0x40b5b40,0x42d6d30
    ConstraintPointOnLine * c110=new ConstraintPointOnLine();
    c110->pvec.push_back(plist_[94]);
    c110->pvec.push_back(plist_[95]);
    c110->pvec.push_back(clist_params_[4]);
    c110->pvec.push_back(clist_params_[5]);
    c110->pvec.push_back(clist_params_[6]);
    c110->pvec.push_back(clist_params_[7]);
    c110->origpvec=c110->pvec;
    c110->rescale();
    clist_.push_back(c110); // addresses = 0x40b6590,0x42d78e0,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(-40)); // 15 address: 0x42f3f80
    clist_.push_back(new ConstraintDifference(clist_params_[1],plist_[95],clist_params_[15])); // addresses = 0x4200ce0,0x42d78e0,0x42f3f80
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(-20)); // 16 address: 0x42f3e70
    clist_.push_back(new ConstraintDifference(clist_params_[1],plist_[89],clist_params_[16])); // addresses = 0x4200ce0,0x42d73b0,0x42f3e70
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(16)); // 17 address: 0x42f4560
    clist_.push_back(new ConstraintEqual(plist_[27],clist_params_[17])); // addresses = 0x4178970,0x42f4560
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(16)); // 18 address: 0x42f4760
    clist_.push_back(new ConstraintEqual(plist_[76],clist_params_[18])); // addresses = 0x41f4970,0x42f4760
    ConstraintPerpendicular * c115=new ConstraintPerpendicular();
    c115->pvec.push_back(plist_[91]);
    c115->pvec.push_back(plist_[92]);
    c115->pvec.push_back(plist_[88]);
    c115->pvec.push_back(plist_[89]);
    c115->pvec.push_back(clist_params_[4]);
    c115->pvec.push_back(clist_params_[5]);
    c115->pvec.push_back(clist_params_[6]);
    c115->pvec.push_back(clist_params_[7]);
    c115->origpvec=c115->pvec;
    c115->rescale();
    clist_.push_back(c115); // addresses = 0x41c6ce0,0x4186b50,0x42d73d0,0x42d73b0,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    ConstraintMidpointOnLine * c116=new ConstraintMidpointOnLine();
    c116->pvec.push_back(plist_[91]);
    c116->pvec.push_back(plist_[92]);
    c116->pvec.push_back(plist_[88]);
    c116->pvec.push_back(plist_[89]);
    c116->pvec.push_back(clist_params_[4]);
    c116->pvec.push_back(clist_params_[5]);
    c116->pvec.push_back(clist_params_[6]);
    c116->pvec.push_back(clist_params_[7]);
    c116->origpvec=c116->pvec;
    c116->rescale();
    clist_.push_back(c116); // addresses = 0x41c6ce0,0x4186b50,0x42d73d0,0x42d73b0,0x42edee0,0x42ee2e0,0x42edec0,0x42ee300
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(140)); // 19 address: 0x40c51e0
    ConstraintP2PDistance * c117=new ConstraintP2PDistance();
    c117->pvec.push_back(plist_[91]);
    c117->pvec.push_back(plist_[92]);
    c117->pvec.push_back(plist_[88]);
    c117->pvec.push_back(plist_[89]);
    c117->pvec.push_back(clist_params_[19]);
    c117->origpvec=c117->pvec;
    c117->rescale();
    clist_.push_back(c117); // addresses = 0x41c6ce0,0x4186b50,0x42d73d0,0x42d73b0,0x40c51e0
    clist_.push_back(new ConstraintEqual(plist_[93],plist_[90])); // addresses = 0x4186b70,0x42d7730
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(15)); // 20 address: 0x40c2a40
    clist_.push_back(new ConstraintEqual(plist_[90],clist_params_[20])); // addresses = 0x42d7730,0x40c2a40
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(12)); // 21 address: 0x40c5240
    clist_.push_back(new ConstraintEqual(plist_[96],clist_params_[21])); // addresses = 0x42d7900,0x40c5240
    clist_.push_back(new ConstraintEqual(plist_[101],plist_[91])); // addresses = 0x41fc0e0,0x41c6ce0
    clist_.push_back(new ConstraintEqual(plist_[102],plist_[92])); // addresses = 0x41865e0,0x4186b50
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(25)); // 22 address: 0x41796f0
    clist_.push_back(new ConstraintEqual(plist_[103],clist_params_[22])); // addresses = 0x4186600,0x41796f0
    clist_.push_back(new ConstraintEqual(plist_[108],plist_[14])); // addresses = 0x42d8d70,0x41f6830
    clist_.push_back(new ConstraintEqual(plist_[109],plist_[15])); // addresses = 0x42d8d90,0x41f6850
    clist_.push_back(new ConstraintEqual(plist_[106],plist_[97])); // addresses = 0x42d8910,0x41fc060
    clist_.push_back(new ConstraintEqual(plist_[107],plist_[98])); // addresses = 0x42d88f0,0x41fc080
    ConstraintTangentCircumf * c128=new ConstraintTangentCircumf(false);
    c128->pvec.push_back(plist_[16]);
    c128->pvec.push_back(plist_[17]);
    c128->pvec.push_back(plist_[110]);
    c128->pvec.push_back(plist_[111]);
    c128->pvec.push_back(plist_[18]);
    c128->pvec.push_back(plist_[112]);
    c128->origpvec=c128->pvec;
    c128->rescale();
    clist_.push_back(c128); // addresses = 0x41f6870,0x40d90c0,0x42d8db0,0x42d8dd0,0x40d90e0,0x42d8df0
    ConstraintTangentCircumf * c129=new ConstraintTangentCircumf(false);
    c129->pvec.push_back(plist_[101]);
    c129->pvec.push_back(plist_[102]);
    c129->pvec.push_back(plist_[110]);
    c129->pvec.push_back(plist_[111]);
    c129->pvec.push_back(plist_[103]);
    c129->pvec.push_back(plist_[112]);
    c129->origpvec=c129->pvec;
    c129->rescale();
    clist_.push_back(c129); // addresses = 0x41fc0e0,0x41865e0,0x42d8db0,0x42d8dd0,0x4186600,0x42d8df0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(32)); // 23 address: 0x42d8ed0
    clist_.push_back(new ConstraintEqual(plist_[112],clist_params_[23])); // addresses = 0x42d8df0,0x42d8ed0
    clist_.push_back(new ConstraintEqual(plist_[115],plist_[99])); // addresses = 0x41dd2c0,0x41fc0a0
    clist_.push_back(new ConstraintEqual(plist_[116],plist_[100])); // addresses = 0x41dd2a0,0x41fc0c0
    ConstraintP2LDistance * c133=new ConstraintP2LDistance();
    c133->pvec.push_back(plist_[101]);
    c133->pvec.push_back(plist_[102]);
    c133->pvec.push_back(plist_[115]);
    c133->pvec.push_back(plist_[116]);
    c133->pvec.push_back(plist_[117]);
    c133->pvec.push_back(plist_[118]);
    c133->pvec.push_back(plist_[103]);
    c133->origpvec=c133->pvec;
    c133->rescale();
    clist_.push_back(c133); // addresses = 0x41fc0e0,0x41865e0,0x41dd2c0,0x41dd2a0,0x42e9fe0,0x42ea000,0x4186600
    clist_.push_back(new ConstraintEqual(plist_[123],plist_[94])); // addresses = 0x42ea1d0,0x40b6590
    clist_.push_back(new ConstraintEqual(plist_[124],plist_[95])); // addresses = 0x42ea1f0,0x42d78e0
    clist_.push_back(new ConstraintEqual(plist_[128],plist_[119])); // addresses = 0x42ea940,0x42ea150
    clist_.push_back(new ConstraintEqual(plist_[129],plist_[120])); // addresses = 0x42ea920,0x42ea170
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(22)); // 24 address: 0x42f0310
    clist_.push_back(new ConstraintEqual(plist_[125],clist_params_[24])); // addresses = 0x42ea210,0x42f0310
    ConstraintP2LDistance * c139=new ConstraintP2LDistance();
    c139->pvec.push_back(plist_[123]);
    c139->pvec.push_back(plist_[124]);
    c139->pvec.push_back(plist_[128]);
    c139->pvec.push_back(plist_[129]);
    c139->pvec.push_back(plist_[130]);
    c139->pvec.push_back(plist_[131]);
    c139->pvec.push_back(plist_[125]);
    c139->origpvec=c139->pvec;
    c139->rescale();
    clist_.push_back(c139); // addresses = 0x42ea1d0,0x42ea1f0,0x42ea940,0x42ea920,0x42eaa60,0x3d97780,0x42ea210
    clist_.push_back(new ConstraintEqual(plist_[136],plist_[3])); // addresses = 0x4200e70,0x4161500
    clist_.push_back(new ConstraintEqual(plist_[137],plist_[4])); // addresses = 0x4200e90,0x41e7780
    clist_.push_back(new ConstraintEqual(plist_[132],plist_[117])); // addresses = 0x4200df0,0x42e9fe0
    clist_.push_back(new ConstraintEqual(plist_[133],plist_[118])); // addresses = 0x4200e10,0x42ea000
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(20)); // 25 address: 0x42fb160
    clist_.push_back(new ConstraintEqual(plist_[138],clist_params_[25])); // addresses = 0x4200eb0,0x42fb160
    ConstraintP2LDistance * c145=new ConstraintP2LDistance();
    c145->pvec.push_back(plist_[136]);
    c145->pvec.push_back(plist_[137]);
    c145->pvec.push_back(plist_[115]);
    c145->pvec.push_back(plist_[116]);
    c145->pvec.push_back(plist_[117]);
    c145->pvec.push_back(plist_[118]);
    c145->pvec.push_back(plist_[138]);
    c145->origpvec=c145->pvec;
    c145->rescale();
    clist_.push_back(c145); // addresses = 0x4200e70,0x4200e90,0x41dd2c0,0x41dd2a0,0x42e9fe0,0x42ea000,0x4200eb0
    clist_.push_back(new ConstraintEqual(plist_[143],plist_[134])); // addresses = 0x42ebe00,0x4200e30
    clist_.push_back(new ConstraintEqual(plist_[144],plist_[135])); // addresses = 0x42ebe20,0x4200e50
    clist_.push_back(new ConstraintEqual(plist_[141],plist_[130])); // addresses = 0x42eba20,0x42eaa60
    clist_.push_back(new ConstraintEqual(plist_[142],plist_[131])); // addresses = 0x42eba00,0x3d97780
    ConstraintP2LDistance * c150=new ConstraintP2LDistance();
    c150->pvec.push_back(plist_[145]);
    c150->pvec.push_back(plist_[146]);
    c150->pvec.push_back(plist_[128]);
    c150->pvec.push_back(plist_[129]);
    c150->pvec.push_back(plist_[130]);
    c150->pvec.push_back(plist_[131]);
    c150->pvec.push_back(plist_[147]);
    c150->origpvec=c150->pvec;
    c150->rescale();
    clist_.push_back(c150); // addresses = 0x42ebe40,0x42ebe60,0x42ea940,0x42ea920,0x42eaa60,0x3d97780,0x42ebe80
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(5)); // 26 address: 0x42ebf60
    clist_.push_back(new ConstraintEqual(plist_[147],clist_params_[26])); // addresses = 0x42ebe80,0x42ebf60
    ConstraintTangentCircumf * c152=new ConstraintTangentCircumf(false);
    c152->pvec.push_back(plist_[136]);
    c152->pvec.push_back(plist_[137]);
    c152->pvec.push_back(plist_[145]);
    c152->pvec.push_back(plist_[146]);
    c152->pvec.push_back(plist_[138]);
    c152->pvec.push_back(plist_[147]);
    c152->origpvec=c152->pvec;
    c152->rescale();
    clist_.push_back(c152); // addresses = 0x4200e70,0x4200e90,0x42ebe40,0x42ebe60,0x4200eb0,0x42ebe80
    clist_.push_back(new ConstraintEqual(plist_[150],plist_[79])); // addresses = 0x42ec600,0x42dacf0
    clist_.push_back(new ConstraintEqual(plist_[151],plist_[80])); // addresses = 0x42ec5e0,0x42dacd0
    ConstraintP2LDistance * c155=new ConstraintP2LDistance();
    c155->pvec.push_back(plist_[83]);
    c155->pvec.push_back(plist_[84]);
    c155->pvec.push_back(plist_[150]);
    c155->pvec.push_back(plist_[151]);
    c155->pvec.push_back(plist_[152]);
    c155->pvec.push_back(plist_[153]);
    c155->pvec.push_back(plist_[85]);
    c155->origpvec=c155->pvec;
    c155->rescale();
    clist_.push_back(c155); // addresses = 0x42db110,0x40b5b20,0x42ec600,0x42ec5e0,0x42eca00,0x42eca20,0x40b5b40
    clist_.push_back(new ConstraintEqual(plist_[158],plist_[88])); // addresses = 0x42ecb90,0x42d73d0
    clist_.push_back(new ConstraintEqual(plist_[159],plist_[89])); // addresses = 0x42ecbb0,0x42d73b0
    clist_.push_back(new ConstraintEqual(plist_[156],plist_[152])); // addresses = 0x42ecb50,0x42eca00
    clist_.push_back(new ConstraintEqual(plist_[157],plist_[153])); // addresses = 0x42ecb70,0x42eca20
    clist_.push_back(new ConstraintEqual(plist_[165],plist_[121])); // addresses = 0x42ed6e0,0x42ea190
    clist_.push_back(new ConstraintEqual(plist_[166],plist_[122])); // addresses = 0x42ed700,0x42ea1b0
    clist_.push_back(new ConstraintEqual(plist_[163],plist_[154])); // addresses = 0x42ed300,0x42ecb10
    clist_.push_back(new ConstraintEqual(plist_[164],plist_[155])); // addresses = 0x42ed2e0,0x42ecb30
    ConstraintP2LDistance * c164=new ConstraintP2LDistance();
    c164->pvec.push_back(plist_[158]);
    c164->pvec.push_back(plist_[159]);
    c164->pvec.push_back(plist_[150]);
    c164->pvec.push_back(plist_[151]);
    c164->pvec.push_back(plist_[152]);
    c164->pvec.push_back(plist_[153]);
    c164->pvec.push_back(plist_[160]);
    c164->origpvec=c164->pvec;
    c164->rescale();
    clist_.push_back(c164); // addresses = 0x42ecb90,0x42ecbb0,0x42ec600,0x42ec5e0,0x42eca00,0x42eca20,0x42ecbd0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(25)); // 27 address: 0x42eccb0
    clist_.push_back(new ConstraintEqual(plist_[160],clist_params_[27])); // addresses = 0x42ecbd0,0x42eccb0
    ConstraintTangentCircumf * c166=new ConstraintTangentCircumf(false);
    c166->pvec.push_back(plist_[158]);
    c166->pvec.push_back(plist_[159]);
    c166->pvec.push_back(plist_[167]);
    c166->pvec.push_back(plist_[168]);
    c166->pvec.push_back(plist_[160]);
    c166->pvec.push_back(plist_[169]);
    c166->origpvec=c166->pvec;
    c166->rescale();
    clist_.push_back(c166); // addresses = 0x42ecb90,0x42ecbb0,0x42ed720,0x42ed740,0x42ecbd0,0x42ed760
    ConstraintTangentCircumf * c167=new ConstraintTangentCircumf(false);
    c167->pvec.push_back(plist_[123]);
    c167->pvec.push_back(plist_[124]);
    c167->pvec.push_back(plist_[167]);
    c167->pvec.push_back(plist_[168]);
    c167->pvec.push_back(plist_[125]);
    c167->pvec.push_back(plist_[169]);
    c167->origpvec=c167->pvec;
    c167->rescale();
    clist_.push_back(c167); // addresses = 0x42ea1d0,0x42ea1f0,0x42ed720,0x42ed740,0x42ea210,0x42ed760
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(-12)); // 28 address: 0x42ed840
    clist_.push_back(new ConstraintDifference(clist_params_[1],plist_[146],clist_params_[28])); // addresses = 0x4200ce0,0x42ebe60,0x42ed840
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(25)); // 29 address: 0x42fc8a0
    clist_.push_back(new ConstraintEqual(plist_[169],clist_params_[29])); // addresses = 0x42ed760,0x42fc8a0
    // Address not in System params...rebuilding into clist_params_
    clist_params_.push_back(new double(37.8319)); // 30 address: 0x42fdf00
    ConstraintP2PDistance * c170=new ConstraintP2PDistance();
    c170->pvec.push_back(plist_[115]);
    c170->pvec.push_back(plist_[116]);
    c170->pvec.push_back(plist_[117]);
    c170->pvec.push_back(plist_[118]);
    c170->pvec.push_back(clist_params_[30]);
    c170->origpvec=c170->pvec;
    c170->rescale();
    clist_.push_back(c170); // addresses = 0x41dd2c0,0x41dd2a0,0x42e9fe0,0x42ea000,0x42fdf00    
    std::cout << "Creating the Subsystem" << std::endl;
    
    SubSystem * mysub = new SubSystem(clist_,plistsub_); // this creation is without reduction map
    
    /*std::cout << "Solving using Levenberg Marquardt" << std::endl;
    solve_LM(mysub, false);*/
    
    std::cout << "Solving using DogLeg" << std::endl;
    solve_DL(mysub, false);
    
    
    return 0;
}
