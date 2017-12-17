/* *****************************************************
 * (c) 2012 Particle In Cell Consulting LLC
 * 
 * This document is subject to the license specified in 
 * Starfish.java and the LICENSE file
 * *****************************************************/
package starfish.core.solver;

import jcuda.jcusparse.csrilu02Info;
import jcuda.jcusparse.csrsv2Info;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import starfish.core.common.Vector;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import starfish.core.common.Starfish;
import starfish.core.common.Starfish.Log;
import starfish.core.domain.FieldCollection2D;
import starfish.core.domain.Mesh;
import starfish.core.domain.Mesh.Face;
import starfish.core.domain.Mesh.Node;
import starfish.core.domain.Mesh.NodeType;
//cuda
import jcuda.*;

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ONE;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseStatus.CUSPARSE_STATUS_ZERO_PIVOT;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

/**
 * Finite volume Matrix solver for Ax=b using the Gauss-Seidel method (for now)
 */
public abstract class Solver
{
	protected int lin_max_it;
	protected double lin_tol;
	protected int nl_max_it;
	protected double nl_tol;
	boolean qn; /* quasi-neutral solver? */

	public double kTe0 = 1; /* reference temperature values for boltzman relationship */
	public double den0 = 1e15;  /* reference values of ion density along bottom edge */
	public double phi0 = 0; /* reference values of potential along bottom edge */

	boolean first = true;
	boolean initial_only = false;

    /* gradient class used to store gradient coefficients at control volume
     * edge midpoints (i+1/2,j) (i,j+1/2) and so on*/

	class Gradient
	{
		double Gi[] = new double[6]; /* i derivative */
		double Gj[] = new double[6]; /* j derivative */
		double R;		    /* radius of calculation*/

		int i[] = {-1, -1, -1, -1, -1, -1}; /* actual i/j indexes for electric field computation*/
		int j[] = {-1, -1, -1, -1, -1, -1};
		int n[] = {-1, -1, -1, -1, -1, -1}; /*unknown index for matrix assembly*/
	}

	public class MeshData
	{
		public Mesh mesh;			/*associated mesh*/
		public Matrix Gi;	/*gradient matrix in i direction*/
		public Matrix Gj;	/*gradient matrix in j direction*/
		public Matrix A;
		public Matrix L;	//LU decomposition of matrix A, if available
		public Matrix U;
		public boolean fixed_node[];
		public double b[];
		public double x[];	    /*solution vector*/
	}
	public MeshData mesh_data[];

	/**
	 * sets solver parameters
	 */
	void setLinParams(int lin_max_it, double lin_tol)
	{
	/*some defaults*/
		this.lin_max_it = lin_max_it;
		this.lin_tol = lin_tol;
	}

	void setNLParams(int nl_max_it, double nl_tol)
	{
	/*some defaults*/
		this.nl_max_it = nl_max_it;
		this.nl_tol = nl_tol;
	}

	/*init*/
	public void init()
	{
		ArrayList<Mesh> mesh_list = Starfish.getMeshList();

	/*allocate memory*/
		mesh_data = new MeshData[mesh_list.size()];

	/*for now*/
		for (int m = 0; m < mesh_list.size(); m++)
		{
			Mesh mesh = Starfish.getMeshList().get(m);
			MeshData md = new MeshData();
			mesh_data[m] = md;
			md.mesh = mesh;

	    /* setup up coefficient matrix */
			int ni = mesh.ni;
			int nj = mesh.nj;
			int nu = ni*nj;

	    /* initialize coefficient matrix */
			md.A = new Matrix(nu);

	    /* initialize array of fixed (dirichlet) nodes*/
			md.fixed_node = new boolean[nu];

			for (int i = 0; i<nu;i++)
				md.fixed_node[i] = false;

	    /*init coefficients*/
			initCoefficients(md);
		}
	}

	public void exit()
	{
	}

	/* initializes matrix coefficients */
	private void initCoefficients(MeshData md)
	{
		int i, j;
		int ni = md.mesh.ni;
		int nj = md.mesh.nj;
		int nu = ni*nj;

	/* *** 1) set coefficients for electric field ******/
		md.Gi = new Matrix(nu);
		md.Gj = new Matrix(nu);

		for (i = 0; i < ni; i++)
			for (j = 0; j < nj; j++)
			{
				Gradient G = CalculateGradient(md, i, j);

		/*assemble non-zero values into gradient matrix*/
				int u = md.mesh.IJtoN(i,j);
				for (int v = 0; v < 6 && G.i[v]>=0; v++)
				{
					if (G.Gi[v]!=0) md.Gi.add(u, G.n[v], G.Gi[v]);
					if (G.Gj[v]!=0) md.Gj.add(u, G.n[v], G.Gj[v]);
				}
			}




		/***** 2) set coefficients for potential solver **************/
		for (i = 0; i < ni; i++)
			for (j = 0; j < nj; j++)
				CalculateCoefficients(md, i, j);

//	for (int u=0;u<ni*nj;u++)
//	    	md.A.println(u,md.mesh.nj);
	}

	/**updates potential and calculates new electric field*/
	abstract public void update();

	public interface NL_Eval
	{

		public double[] eval_b(double x[], boolean fixed[]);

		public double[] eval_prime(double x[], boolean fixed[]);
	}

	/**
	 * solves the nonlinear problem Ax-b=0 using the Newtons method
	 * This method is based on the algorithm on page 614 in Numerical Analysis
	 * F(x)=Ax-b
	 * J(x)=dF/dx=A-db/dx=A-P
	 * solve Jy=F
	 * update x=x-y
	 * also b=b0+b(x) where b0 is constant
	 */
	public int solveNonLinearNR(MeshData md[], NL_Eval nl_eval)
	{
		int it;
		double norm=-1;

		boolean fixed_node[] = md[0].fixed_node;
		double x[] = md[0].x;
		double b0[] = md[0].b;
		Matrix A = md[0].A;

		double y[] = new double[x.length];

		MeshData md_nl[] = new MeshData[1];
		md_nl[0] = new MeshData();

		for (it = 0; it < nl_max_it; it++)
		{
	    /*calculate b(x)*/
			double b_x[] = nl_eval.eval_b(x, fixed_node);

	    /*rhs: b=b0+b_x*/
			double b[]= Vector.add(b0, b_x);

	    /*calculate P(x) = db/dx*/
			double P[] = nl_eval.eval_prime(x, fixed_node);

	    /*TEMPORARY*/
			Mesh mesh = md[0].mesh;
	    /*update boundaries */
			for (int j=0;j<mesh.nj;j++)
			{
				b[mesh.IJtoN(0, j)] = mesh.node[0][j].bc_value;
				b[mesh.IJtoN(mesh.ni-1, j)] = mesh.node[mesh.ni-1][j].bc_value;

				P[mesh.IJtoN(0, j)] = 0;
				P[mesh.IJtoN(mesh.ni-1, j)] = 0;
			}

			for (int i=0;i<mesh.ni;i++)
			{
				b[mesh.IJtoN(i, 0)] = mesh.node[i][0].bc_value;
				b[mesh.IJtoN(i,mesh.nj-1)] = mesh.node[i][mesh.nj-1].bc_value;
				P[mesh.IJtoN(i, 0)] = 0;
				P[mesh.IJtoN(i,mesh.nj-1)] = 0;
			}

	    /*calculate F(x)=Ax-b*/
			double F[] = Vector.subtract(A.mult(x), b);

	    /*calculate J(x) = d/dx(Ax-b) = A-diag(P)*/
			Matrix J = A.subtractDiag(P);

	    /*solve Jy=F*/
	    /*temporary hack*/
			md_nl[0].A=J;
			md_nl[0].b=F;
			md_nl[0].mesh= md[0].mesh;
			md_nl[0].fixed_node = md[0].fixed_node;
			md_nl[0].x=y;

			int lin_it = solveLinearGS(md_nl);

	    /*set x=x-y*/
			Vector.subtractInclusive(x, y);

	    /*check norm(y) for convergence*/
			norm = Vector.norm(y);
			//  System.out.println(b0[mesh.IJtoN(5,10)]+" "+ lin_it+" "+norm);

			Log.log(">>>>NR:" + it + " " + String.format("%.2g", norm));
			if (norm < nl_tol)
			{
				break;
			}
		}

		if (it==nl_max_it)
			Log.warning("!! NR solver failed to converge, norm = "+norm);
		return it;
	}

	/**
	 * solves Ax=b for x using the GS method
	 */
	public int solveLinearGS(MeshData mesh_data[])
	{
		double norm = lin_tol;
		long start, stop;
		double time_solve;

	/*create threads*/
		int np = 1;
		//cuda
		ParLinearGS p = null;

		Collection<Callable<String>> workers = new ArrayList<Callable<String>>();
		ExecutorService executor = Executors.newFixedThreadPool(1);

		for (MeshData md:mesh_data)
		{
			Matrix A = md.A;
			double x[] = md.x;
			double b[] = md.b;

			int nu = x.length;
			int i_d = (int)((double)nu/np)+1;
			int i_min=0;
			//cuda
			for (int i=0, id=0;i<np;i++,i_min+=i_d,id++)
				//workers.add(new ParLinearGS(id,i_min,i_min+i_d,A,x,b));
				p = new ParLinearGS(id,i_min,i_min+i_d,A,x,b);

			/*
			i_d = (int)((double)nu/np)+1;
			i_min=0;
			for (int i=0, id=0;i<np;i++,i_min+=i_d,id++)
				workers.add(new ParLinearGS(id,i_min,i_min+i_d,A,x,b));*/
		}

	/* SOLVER */
		int it = 1;			/*start with one so we don't compute residue on first run*/
		start = System.nanoTime();
		while (it <= lin_max_it)
		{
			try
			{
				executor.invokeAll(workers);
				//cuda
				p.call_cuda();
			} catch (Exception ex)
			{
				Logger.getLogger(Solver.class.getName()).log(Level.SEVERE, null, ex);
			}

			/*** update boundaries**/

	    /*inflate solution*/
			for (MeshData md:mesh_data)
				Vector.inflate(md.x, md.mesh.ni, md.mesh.nj, Starfish.domain_module.getPhi(md.mesh).getData());

			Starfish.domain_module.getPhi().syncMeshBoundaries();

			for (MeshData md:mesh_data)
			{
				Mesh mesh = md.mesh;

			/*flatten data*/
				Vector.deflate(Starfish.domain_module.getPhi(mesh).getData(),md.x);

		/*add objects*/
				Vector.merge(md.fixed_node, md.x, md.b, md.b);

			}

	/* check convergence */
			if (it % 25 == 0)
			{
				norm=0;
				int nn=0;

				for (MeshData md:mesh_data)
				{
					norm += calculateResidue(md.A, md.x, md.b);
					nn += md.x.length;
				}

				norm/=nn;

				//System.out.println(norm);
				if (norm < lin_tol)
					break;
			}
			it++;
		}
		it--;
		if (it >= lin_max_it)
		{
			Log.warning(" !! GS failed to converge in " + it + " iteration, norm = " + norm);
		}
		stop = System.nanoTime();
		time_solve = (stop - start) / 1e9;
		System.out.printf("timing: = %10.6f sec\n", time_solve);
		//executor.shutdown();
		//cuda
		p.freeMemory();
		return it;
	}

	class ParLinearGS<K> implements Callable
	{
		int i_min,i_max;
		Matrix A;
		double x[],b[];
		protected int id;

		//cuda
		ArrayList<MatrixElement> A_temp;
		int cooRowIndexHostPtr[];
		int cooColIndexHostPtr[];
		double cooValHostPtr[];
		double tauHost[];
		Pointer cooRowIndex;
		Pointer cooColIndex;
		Pointer cooVal;
		Pointer csrRowPtr;
		cusparseHandle handle;
		cusparseMatDescr descra;
		int l, x_l;
		Pointer t;
		Pointer xPtr;


		ParLinearGS(int id, int i_min, int i_max,Matrix A, double x[], double b[])
		{
			this.id=id;
			this.i_min=i_min;
			this.i_max=i_max<A.nr?i_max:A.nr;
			//Matrix temp_A = new Matrix(3);
			/*temp_A.set(0,0, 3.5454);
			temp_A.set(0,1, 2.5252);
			temp_A.set(0,2, 4.295295292952952529295);
			temp_A.set(1,0, 4.52952592);
			temp_A.set(1,1, 5.25214141171717);
			temp_A.set(1,2, 8.14818184184);
			temp_A.set(2,0, 2.41841891812841284);
			temp_A.set(2,1, 2.982981241298292929);
			temp_A.set(2,2, 7.52925294141962592);
			this.A=temp_A;*/
			this.x=x;
			this.A = A;
			/*double[] x_temp = new double[3];
			x_temp[0] = 2.2525267479497494974794717197197419714971971;
			x_temp[1] = 0.000000000000000000000000000001;
			x_temp[2] = 9295259292429819412414.0;
			this.x = x_temp;*/
			this.b=b;

			//cuda
			cooRowIndex = new Pointer();
			cooColIndex = new Pointer();
			cooVal = new Pointer();
			csrRowPtr = new Pointer();
			t = new Pointer();
			xPtr = new Pointer();

			A_temp = new ArrayList<MatrixElement>();
			int r, c;
			double val;
			for(int i = 0; i < this.A.data.size(); i++){
				for(Map.Entry<Integer, Double> it : this.A.data.get(i).entrySet()){
					if(it != null){
						r = i;
						c = it.getKey();
						if(r != c){
							val = it.getValue();
							MatrixElement el = new MatrixElement(r, c, val);
							A_temp.add(el);
						}
					}
				}
			}

			l = A_temp.size();
			x_l = x.length;

			handle = new cusparseHandle();
			descra = new cusparseMatDescr();
			cooRowIndexHostPtr = new int[l];
			cooColIndexHostPtr = new int[l];
			cooValHostPtr      = new double[l];
			tauHost	 = new double[x_l];

			for(int i = 0; i < l; i++){
				cooRowIndexHostPtr[i] = A_temp.get(i).row;
				cooColIndexHostPtr[i] = A_temp.get(i).column;
				cooValHostPtr[i] = A_temp.get(i).value;
			}

			cudaMalloc(cooRowIndex, l*Sizeof.INT);
			cudaMalloc(cooColIndex, l*Sizeof.INT);
			cudaMalloc(cooVal,      l*Sizeof.DOUBLE);
			cudaMalloc(xPtr, 		x_l*Sizeof.DOUBLE);
			cudaMalloc(t,			x_l*Sizeof.DOUBLE);


/*			for(int u = 0; u < 3; u++){
				double temp_tau = this.A.multRowNonDiag(this.x, u);
				System.out.println(temp_tau);
			}
			/*cooRowIndexHostPtr[0] = 0; cooColIndexHostPtr[0] = 1; cooValHostPtr[0] = 2;
			cooRowIndexHostPtr[1] = 0; cooColIndexHostPtr[1] = 2; cooValHostPtr[1] = 4;
			cooRowIndexHostPtr[2] = 1; cooColIndexHostPtr[2] = 0; cooValHostPtr[2] = 4;
			cooRowIndexHostPtr[3] = 1; cooColIndexHostPtr[3] = 2; cooValHostPtr[3] = 8;
			cooRowIndexHostPtr[4] = 2; cooColIndexHostPtr[4] = 0; cooValHostPtr[4] = 2;
			cooRowIndexHostPtr[5] = 2; cooColIndexHostPtr[5] = 1; cooValHostPtr[5] = 2;*/

			cudaMemcpy(cooColIndex, Pointer.to(cooColIndexHostPtr), l*Sizeof.INT,          cudaMemcpyHostToDevice);
			cudaMemcpy(cooRowIndex, Pointer.to(cooRowIndexHostPtr), l*Sizeof.INT,          cudaMemcpyHostToDevice);
			cudaMemcpy(cooVal,      Pointer.to(cooValHostPtr),      l*Sizeof.DOUBLE,        cudaMemcpyHostToDevice);
			cudaMemcpy(xPtr,		Pointer.to(this.x), 					x_l*Sizeof.DOUBLE, 	cudaMemcpyHostToDevice);

			cusparseCreate(handle);
			cusparseCreateMatDescr(descra);
			cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);
			cudaMalloc(csrRowPtr, (this.A.nr+1)*Sizeof.INT);
			cusparseXcoo2csr(handle, cooRowIndex, l, this.A.nr,
					csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);

			/*int stat = cusparseDcsrmv(
					handle, CUSPARSE_OPERATION_NON_TRANSPOSE, A.nr, A.nr, l,
					Pointer.to(new double[]{1.0f}), descra, cooVal, csrRowPtr,
					cooColIndex, xPtr, Pointer.to(new double[]{0.0f}), t);

			System.out.println(stat);
			stat = cudaMemcpy(Pointer.to(tauHost), t, x_l*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			System.out.println(stat);
			System.out.println("wynik:");
			for(int i = 0 ; i < 3; i++)
				System.out.println(tauHost[i]);*/
		}

		@Override
		public String call() throws Exception
		{
			for (int u=i_min;u<i_max;u++)
			{
		/*tau = [A-D]x */

				double tau = A.multRowNonDiag(x, u);

				double g = (b[u] - tau) / A.get(u,u);

				x[u] = x[u] + 1.4*(g-x[u]); /*SOR*/
			}

			return null;
		}

		public String call_cuda() throws Exception
		{
			double[] alfa = new double[1];
			alfa[0] = 1;
			/*double[] tmp = new double[4141];
			for (int i = 0 ; i < 4141; ++i)
				tmp[i] = 1;
			Pointer tmpD = new Pointer();
			cudaMalloc(tmpD, 4141 * Sizeof.DOUBLE);
			cudaMemcpy(tmpD, Pointer.to(tmp), 4141*Sizeof.DOUBLE, cudaMemcpyHostToDevice);*/

			int stat = cusparseDcsrmv(
					handle, CUSPARSE_OPERATION_NON_TRANSPOSE, A.nr, A.nr, l,
					Pointer.to(alfa), descra, cooVal, csrRowPtr,
					cooColIndex, xPtr, Pointer.to(new double[]{0.0f}), t);

			stat = cudaMemcpy(Pointer.to(tauHost), t, x_l*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

			for (int u=i_min;u<i_max;u++)
			{
		/* tau = [A-D]x */

				double g = (b[u] - tauHost[u]) / A.get(u,u);

				x[u] = x[u] + 1.4*(g-x[u]); /*SOR*/
			}
			cudaMemcpy(xPtr, Pointer.to(x),x_l*Sizeof.DOUBLE, cudaMemcpyHostToDevice);

			return null;
		}

		public void freeMemory(){
			cudaFree(this.t);
			cudaFree(this.csrRowPtr);
			cudaFree(this.xPtr);
			cudaFree(this.cooColIndex);
			cudaFree(this.cooRowIndex);
			cudaFree(this.cooVal);

			cusparseDestroy(this.handle);
		}

	}

	/**
	 * solves Ax=b for x using the Multigrid method
	 */
	protected int solveLinearMultigrid(Matrix A, double x[], double b[])
	{
	/*TODO: Implement*/
		throw new UnsupportedOperationException("Not yet implemented");
	}

	protected int solveLU(MeshData mesh_data[])
	{
		ArrayList<MatrixElement> A_temp;
		int cooRowIndexHostPtr[];
		int cooColIndexHostPtr[];
		double cooValHostPtr[];
		Pointer cooRowIndex;
		Pointer cooColIndex;
		Pointer cooVal;
		Pointer csrRowPtr;
		Pointer pbuffer;
		int[] pbuffer_A = {0};
		int[] pbuffer_L = {0};
		int[] pbuffer_U = {0};
		int[] pbuffer_size = {0};
		cusparseHandle handle;
		cusparseMatDescr descr_A, descr_L, descr_U;
		int l, n;
		for (MeshData md:mesh_data) {
			if (md.L == null || md.U == null) {
				//try {
				//cuda
				/*Matrix A_test = new Matrix(3);
				A_test.set(0, 0, 1.0);
				A_test.set(0, 1, 1.0);
				A_test.set(1, 0, -1.0);
				A_test.set(1, 1, 1.0);
				A_test.set(2, 2, -0.5);*/
				A_temp = new ArrayList<MatrixElement>();
				int r, c;
				double val;
				for (int i = 0; i < md.A.data.size(); i++) {
					for (Map.Entry<Integer, Double> it : md.A.data.get(i).entrySet()) {
						if (it != null) {
							r = i;
							c = it.getKey();
							val = it.getValue();
							MatrixElement el = new MatrixElement(r, c, val);
							A_temp.add(el);

						}
					}
				}
				n = md.A.nr;
				l = A_temp.size();

				cooRowIndex = new Pointer();
				cooColIndex = new Pointer();
				cooVal = new Pointer();
				csrRowPtr = new Pointer();
				pbuffer = new Pointer();
				handle = new cusparseHandle();
				descr_A = new cusparseMatDescr();
				descr_L = new cusparseMatDescr();
				descr_U = new cusparseMatDescr();
				cooRowIndexHostPtr = new int[l];
				cooColIndexHostPtr = new int[l];
				cooValHostPtr = new double[l];

				for (int i = 0; i < l; i++) {
					cooRowIndexHostPtr[i] = A_temp.get(i).row;
					cooColIndexHostPtr[i] = A_temp.get(i).column;
					cooValHostPtr[i] = A_temp.get(i).value;
				}

				int statusm = cudaMalloc(cooRowIndex, l * Sizeof.INT);
				System.out.println("malloc row: "+statusm);
				cudaMalloc(cooColIndex, l * Sizeof.INT);
				cudaMalloc(cooVal, l * Sizeof.DOUBLE);

				statusm = cudaMemcpy(cooColIndex, Pointer.to(cooColIndexHostPtr), l * Sizeof.INT, cudaMemcpyHostToDevice);
				System.out.println("memcpy col: "+statusm);
				cudaMemcpy(cooRowIndex, Pointer.to(cooRowIndexHostPtr), l * Sizeof.INT, cudaMemcpyHostToDevice);
				cudaMemcpy(cooVal, Pointer.to(cooValHostPtr), l * Sizeof.DOUBLE, cudaMemcpyHostToDevice);

				//create handle
				cusparseCreate(handle);
				//set matrix descriptions
				cusparseCreateMatDescr(descr_A);
				cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

				cusparseCreateMatDescr(descr_L);
				cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
				cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
				cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

				cusparseCreateMatDescr(descr_U);
				cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
				cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
				cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);


				//conversion coo to csr
				cudaMalloc(csrRowPtr, (n + 1) * Sizeof.INT);
				cusparseXcoo2csr(handle, cooRowIndex, l, n,
						csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);

				csrilu02Info info_A = new csrilu02Info();
				csrsv2Info info_L = new csrsv2Info();
				csrsv2Info info_U = new csrsv2Info();

				cusparseCreateCsrilu02Info(info_A);
				cusparseCreateCsrsv2Info(info_L);
				cusparseCreateCsrsv2Info(info_U);

				//buffer size

				int status = cusparseDcsrilu02_bufferSize(
						handle, n, l, descr_A, cooVal, csrRowPtr, cooColIndex,
						info_A, pbuffer_A);
				System.out.println("buffersize lu: " + status);
				status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, l, descr_L, cooVal, csrRowPtr, cooColIndex, info_L,
						pbuffer_L);
				System.out.println("buffersize l: " + status);
				status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, l, descr_U, cooVal, csrRowPtr, cooColIndex, info_U,
						pbuffer_U);
				System.out.println("buffersize u:"+ status);

				pbuffer_size[0] = Math.max(pbuffer_A[0], Math.max(pbuffer_L[0], pbuffer_U[0]));
				cudaMalloc(pbuffer, Sizeof.DOUBLE*pbuffer_size[0]);
				System.out.println(pbuffer_size[0]);
				//analysis
				status = cusparseDcsrilu02_analysis(handle, n, l, descr_A,
						cooVal, csrRowPtr, cooColIndex, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
						pbuffer);
				System.out.println("status analysis lu: "+status);
				status = cudaDeviceSynchronize();
				System.out.println("1: "+status);
				status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
						l, descr_L, cooVal, csrRowPtr, cooColIndex, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
						pbuffer);
				System.out.println("analysis l: " + status);
				status = cudaDeviceSynchronize();
				System.out.println("2 : "+status);


				status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
						l, descr_U, cooVal, csrRowPtr, cooColIndex, info_U, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
						pbuffer);
				System.out.println("analysis u: "+status);
				status = cudaDeviceSynchronize();
				System.out.println("3: "+status);

				//solve A~LU
				status = cusparseDcsrilu02(handle,n, l, descr_A, cooVal, csrRowPtr, cooColIndex,
						info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pbuffer);
				System.out.println("status solve: "+status);
				status = cudaDeviceSynchronize();
				System.out.println("solve lu: "+status);

				//Matrix ret[] = md.A.decomposeLU();
				//md.L = ret[0]; md.U = ret[1];
				/*}
				catch (UnsupportedOperationException e)
				{
					Log.error("LU decomposition failed");
				}*/
				//}

				double b[] = md.b;
				double x[] = md.x;
				//Matrix L = md.L;
				//Matrix U = md.U;

				//first solve Ly=b using forward subsitution
				//int n = md.A.nr;

				/*double b[] = new double[n];
				double x[] = new double[n];
				b[0] = 1.0;
				b[1] = 1.0;
				b[2] = 1.0;*/
			/*double y[] = new double[n];
			y[0] = b[0] / L.get(0,0);
			for (int i=1;i<n;i++)
			{
				double s = b[i];
				for (int j=0;j<i;j++)
					s -= L.get(i,j) * y[j];
				y[i] = s/L.get(i, i);
			}

			//now solve Ux=y
			x[n-1] = y[n-1]/U.get(n-1,n-1);
			for (int i=n-2;i>=0;i--)
			{
				double s = y[i];
				for (int j=i+1;j<n;j++)
					s -= U.get(i,j) * x[j];
				x[i] = s/U.get(i,i);
			}*/

				double y[] = new double[n];

				Pointer y_Ptr = new Pointer();
				Pointer b_Ptr = new Pointer();
				Pointer x_Ptr = new Pointer();

				status = cudaMalloc(b_Ptr, n*Sizeof.DOUBLE);
				System.out.println("malloc b: "+status);
				status = cudaMalloc(y_Ptr, n*Sizeof.DOUBLE);
				System.out.println("malloc y: "+status);
				status=cudaMalloc(x_Ptr, n*Sizeof.DOUBLE);
				System.out.println("malloc x: "+status);

				status = cudaDeviceSynchronize();
				System.out.println("synchronize solve lu: "+status);
				double[] temp_result = new double[l];
				cudaMemcpy(Pointer.to(temp_result), cooVal, l, cudaMemcpyDeviceToHost);
				System.out.println(temp_result);
				status = cudaMemcpy(b_Ptr, Pointer.to(b), n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
				System.out.println("memcpy b: "+status);
				status = cudaDeviceSynchronize();
				System.out.println("synchronize memcpy b: "+status);
				//Ly = b
				status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, l,
						Pointer.to(new double[]{1.0f}), descr_L, cooVal, csrRowPtr, cooColIndex,
						info_L, b_Ptr, y_Ptr, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pbuffer);
				System.out.println("solve ly = b : "+status);
				status = cudaDeviceSynchronize();
				System.out.println("synchronize ly=b: "+status);

				//Ux=y
				status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, l,
						Pointer.to(new double[]{1.0f}), descr_U, cooVal, csrRowPtr, cooColIndex,
						info_U, y_Ptr, x_Ptr, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pbuffer);
				System.out.println("solve ux=y: "+status);
				status = cudaDeviceSynchronize();
				System.out.println("synchronize ux=y: "+status);

				//copy result to x
				status = cudaMemcpy(Pointer.to(x), x_Ptr, n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
				System.out.println("memcpy result" + status);
				System.out.println("result" + x);
				//free
				cudaFree(pbuffer);
				cusparseDestroyMatDescr(descr_A);
				cusparseDestroyMatDescr(descr_L);
				cusparseDestroyMatDescr(descr_U);
				cusparseDestroyCsrilu02Info(info_A);
				cusparseDestroyCsrsv2Info(info_L);
				cusparseDestroyCsrsv2Info(info_U);
				cusparseDestroy(handle);

			}
		}

		return 0;
	}

	/**
	 * solves Ax=b for x using the PCG method*/
	protected int solveLinearPCG(MeshData md[])
	{
		double norm = lin_tol;
		double alpha;
		double beta;

		double b[][] = new double[md.length][];
		double x[][] = new double[md.length][];
		Matrix A[]= new Matrix[md.length];
		Matrix Pi[] = new Matrix[md.length];
		double r[][] = new double[md.length][];
		double d[][] = new double[md.length][];
		double del_new[] = new double[md.length];

		for (int i=0;i<md.length;i++)
		{
			b[i]= md[i].b;
			x[i]= md[i].x;
			A[i] = md[i].A;
	 
	    /*diagonal preconditioner*/
			b[i] = Vector.mult(b[i], -1);
			for (int j = 0; j < A[i].nr; j++)
				A[i].multRow(j, -1);

			//Matrix M = Matrix.diag_matrix(A);
			Matrix P = A[i].identity();
			Pi[i] = P.inverse();
			//	double Mi[] = Matrix.diag(Matrix.inverse(M));

	    /*initialize*/
			r[i] = Vector.subtract(b[i], A[i].mult(x[i]));
			d[i] = Pi[i].mult(r[i]);
			del_new[i] = Vector.dot(r[i], d[i]);
		}

	/* SOLVER */
		int it = 1;			/*start with one so we don't compute residue on first run*/
		while (it <= lin_max_it)
		{
	    /*reset norm*/
			norm = -1;
	    
	    /*iterate over all meshes*/
			for (int i=0;i<md.length;i++)
			{
				double q[] = A[i].mult(d[i]);

				double t = Vector.dot(d[i], q);
				if (t == 0.0)
				{
		    /*already at exact solution*/
					continue;
				}
				alpha = del_new[i] / t;

				Vector.addInclusive(x[i], Vector.mult(d[i], alpha));
				Vector.subtractInclusive(r[i], Vector.mult(q, alpha));
				//r = Vector.subtract(b, Matrix.mult(A,x));

				double s[] = Pi[i].mult(r[i]);
				double del_old = del_new[i];
				del_new[i] = Vector.dot(r[i], s);
				beta = del_new[i] / del_old;
				d[i] = Vector.add(s, Vector.mult(d[i], beta));
		
		/* check convergence */
				if (it % 25 == 0)
				{
					double norm_l = Vector.norm(r[i]);
					if (norm_l>norm) norm=norm_l;
				}
			}

			//	System.out.println(norm);
			if (norm>0 && norm < lin_tol)
			{
				break;
			}
			it++;
			System.out.printf("it: %d, norm: %g\n",it,norm);
		}

		it--;
		if (it >= lin_max_it)
		{
			Log.warning(" !! PCG failed to converge in " + it + " iteration, norm = " + norm);
		}

		return it;
	}

	/**
	 * returns L2 norm of R=|Ax-b|
	 */
	protected double calculateResidue(Matrix A, double x[], double b[])
	{
	/*this is ||Ax-b||*/
		double norm = Vector.norm(Vector.subtract(A.mult(x), b));
		if (Double.isInfinite(norm)
				|| Double.isNaN(norm))
		{
			Log.error("Solver diverged, aborting");
		}

		return norm;
	}

	/**
	 * Calculates the FVM coefficients at node i,j
	 */
	void CalculateCoefficients(MeshData md, int i0, int j0)
	{
		double[] x1, x2;
		double i, j;
		double R;
		int ni = md.mesh.ni;
		int nj = md.mesh.nj;

		Node node[][] = md.mesh.getNodeArray();

		i = i0;
		j = j0;	/*initialize*/
		int u = md.mesh.IJtoN(i0, j0);

		double phi[][] = Starfish.domain_module.getPhi(md.mesh).getData();

	/*check for fixed nodes*/
		if (node[i0][j0].type == NodeType.DIRICHLET ||
				node[i0][j0].type == NodeType.MESH)
		{
	    /*set phi*/
			phi[i0][j0] = node[i0][j0].bc_value;
			md.A.clearRow(u);
			md.A.set(u,u, 1);	    /*set one on the diagonal*/
			md.fixed_node[u] = true;
			return;
		}
	
	/*boundary nodes*/
		if (i0==0 || i0==ni-1) {md.A.copyRow(md.Gi,u); return;}
		if (j0==0 || j0==nj-1) {md.A.copyRow(md.Gj,u); return;}
	

	/*not a fixed node*/
		for (Face face : Face.values())
		{
			EdgeData e = ComputeEdgeData(face,i0,j0,md);

			Gradient G = CalculateGradient(md, e.im, e.jm);
			R = G.R;
		
	    /*contribution along each face is grad(phi)* ndA, normal vector given by <dj, -di>*/
			MultiplyCoeffs(G.Gi, e.dj_ni * R);
			MultiplyCoeffs(G.Gj, e.di_nj * R);

			for (int v = 0; v < 6 && G.i[v]>=0; v++)
			{
				double val = G.Gj[v]+G.Gi[v];
				if (val!=0) md.A.add(u,G.n[v],val);
			}
	    
	    
    	    /* else if (node[i0][j0].type == NodeType.OPEN
		    || node[i0][j0].type == NodeType.SYMMETRY)
	    {
			R = md.mesh.R(i0,j0);
			node[i0][j0].bc_value*=-R;
	    }*/
		}

	/* scale by volume */
		double V = md.mesh.nodeVol(i0, j0);
		md.A.multRow(u, 1.0 / V);
		node[i0][j0].bc_value /= V;
	}

	/*
     * Returns coefficients for gradient calculation at location i,j 
    i and j can be half indices     */
	private Gradient CalculateGradient(MeshData md, double i0, double j0)
	{
		Gradient G = new Gradient();

		for (Face face : Face.values())
		{
			EdgeData e = ComputeEdgeData(face,i0,j0,md);

	    /* update gradient data */
			for (int n = 0; n < 4; n++)
			{
				for (int k = 0; k < 6; k++)
				{
					if ((G.i[k] == e.N[n][0] && G.j[k] == e.N[n][1]) || G.i[k] < 0)
					{
						G.i[k] = e.N[n][0];			/*set for cases when we are adding new node*/
						G.j[k] = e.N[n][1];
						G.n[k] = md.mesh.IJtoN(G.i[k], G.j[k]);
						G.Gi[k] += 0.25 * e.dj_ni;	/*if all four nodes are the same, then we add 1*/
						G.Gj[k] += 0.25 * e.di_nj;

						break; /* break out of the k loop, go to the next N value*/
					}
				}
			}
		}

	/*scale by area*/
		double A = md.mesh.area(i0, j0);
		MultiplyCoeffs(G.Gj, 1 / A);
		MultiplyCoeffs(G.Gi, 1 / A);

	/*radius at which gradient was calculated*/
		G.R = md.mesh.R(i0, j0);

		return G;
	}

	class EdgeData
	{
		double im, jm;	/*midpoint location*/
		double dj_ni;	/*dj * ni, area * normal vector*/
		double di_nj;	/*di * nj, area * normal vector*/
		int N[][] = new int[4][];
	}

	/*this function computes the control volume face "area" and the mesh points used in evaluation*/
	private EdgeData ComputeEdgeData(Face face, double i0, double j0, MeshData md)
	{
		EdgeData e = new EdgeData();
		int ni = md.mesh.ni;
		int nj = md.mesh.nj;
		double im,jm;	    /*edge mid point*/
		double i1,j1;
		double i2,j2;

		switch (face)
		{
			case RIGHT:
				im = i0 + 0.5;
				jm = j0;
				i1 = im;
				j1 = jm - 0.5;
				i2 = im;
				j2 = jm + 0.5;
				break;
			case TOP:
				im = i0;
				jm = j0 + 0.5;
				i1 = im + 0.5;
				j1 = jm;
				i2 = im - 0.5;
				j2 = jm;
				break;
			case LEFT:
				im = i0 - 0.5;
				jm = j0;
				i1 = im;
				j1 = jm + 0.5;
				i2 = im;
				j2 = jm - 0.5;
				break;
			case BOTTOM:
				im = i0;
				jm = j0 - 0.5;
				i1 = im - 0.5;
				j1 = jm;
				i2 = im + 0.5;
				j2 = jm;
				break;
			default: throw new RuntimeException("Unknown face");
		}

	/*TODO: this is MESH face, need to evaluate cell face*/
		//   NodeType bc_type = md.mesh.mesh_bc[facordinal()].type;

	/*make sure edge end points are in domain*/
		if ((face == Face.LEFT || face == Face.RIGHT)/* && bc_type != NodeTypPERIODIC*/)
		{
			if (j1 < 0)  j1 = 0;
			else if (j1 > nj - 1)  j1 = nj - 1;

			if (j2 < 0)  j2 = 0;
			else if (j2 > nj - 1)  j2 = nj - 1;

			if (im < 0)  im = 0;
			else if (im > ni - 1)  im = ni - 1;
		}

		if ((face == Face.TOP || face == Face.BOTTOM)/* && bc_type != NodeTypPERIODIC*/)
		{
			if (i1 < 0)  i1 = 0;
			else if (i1 > ni - 1)   i1 = ni - 1;

			if (i2 < 0)  i2 = 0;
			else if (i2 > ni - 1)  i2 = ni - 1;

			if (jm < 0)  jm = 0;
			else if (jm > nj - 1)  jm = nj - 1;
		}

	/* compute dA*n (idx-jdy) */
		double x1[] = md.mesh.pos(i1, j1);
		double x2[] = md.mesh.pos(i2, j2);

	/* normal vector is <dx, -dy>*/
		e.dj_ni = x2[1] - x1[1];
		e.di_nj = x1[0] - x2[0];	//multiplied by -1
	
		    /* find neighbor nodes */
		e.N[0] = md.mesh.NodeIndexOffset(im, jm, -0.5, -0.5);
		e.N[1] = md.mesh.NodeIndexOffset(im, jm, 0.5, -0.5);
		e.N[2] = md.mesh.NodeIndexOffset(im, jm, 0.5, 0.5);
		e.N[3] = md.mesh.NodeIndexOffset(im, jm, -0.5, 0.5);

		e.im = im;
		e.jm = jm;

		return e;
	}

	/* This functions multiplies all stencil coefficients by the given value */
	private void MultiplyCoeffs(double[] C, double val)
	{
		for (int i = 0; i < 6; i++)
			C[i] *= val;
	}

	/*function used to compute, for instance, the electric field*/
	public abstract void updateGradientField();

	/*evaluate gradient of data "x" times scale and stores it into fi and fj*/
	protected void computeGradient(double x[], double gi[], double gj[], MeshData md, double scale)
	{
		int i, j, l;

		Mesh mesh = md.mesh;
		md.Gi.mult(x,gi);	/*gi = Gi*x*/
		md.Gj.mult(x,gj);

		if (scale!=1.0)	    /*scale data*/
		{
			Vector.mult(gi, scale, gi);
			Vector.mult(gj, scale, gj);
		}

	}
}
