
extern "C"
__global__ void pushParticle(double *dev_lc0, double *dev_lc1, double *dev_vel0,
														double *dev_vel1, double *dev_dt, double dev_qm,
													double *dev_efi, double *dev_efj, double *dev_bfi, double *dev_bfj,
												int dev_p_transfer, double *dev_ef, double *dev_bf, int *dev_alive, int size,
											double dt, int col)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size && dev_alive[i] == 0){
		if(dev_p_transfer)
			dev_dt[i] += dt;

		double fi = dev_lc0[i];
		double fj = dev_lc1[i];
		int a = int(fi);
		int b = int(fj);

		double di = fi - a;
		double dj = fj - b;

		double v_efi;
		v_efi = (1-di)*(1-dj)*dev_efi[a*col+b];
		v_efi+= di*(1-dj)*dev_efi[(a+1)*col+b];
    v_efi+= di*dj*dev_efi[(a+1)*col +b+1];
    v_efi+= (1-di)*dj*dev_efi[a*col +b+1];

		double v_efj;
		v_efj = (1-di)*(1-dj)*dev_efj[a*col+b];
		v_efj+= di*(1-dj)*dev_efj[(a+1)*col+b];
    v_efj+= di*dj*dev_efj[(a+1)*col +b+1];
    v_efj+= (1-di)*dj*dev_efj[a*col +b+1];

		double v_bfi;
		v_bfi = (1-di)*(1-dj)*dev_bfi[a*col+b];
		v_bfi+= di*(1-dj)*dev_bfi[(a+1)*col+b];
    v_bfi+= di*dj*dev_bfi[(a+1)*col +b+1];
    v_bfi+= (1-di)*dj*dev_bfi[a*col +b+1];

		double v_bfj;
		v_bfj = (1-di)*(1-dj)*dev_bfj[a*col+b];
		v_bfj+= di*(1-dj)*dev_bfj[(a+1)*col+b];
    v_bfj+= di*dj*dev_bfj[(a+1)*col +b+1];
    v_bfj+= (1-di)*dj*dev_bfj[a*col +b+1];

		dev_ef[0] = v_efi;
		dev_ef[1] = v_efj;

		dev_bf[0] = v_bfi;
		dev_bf[1] = v_bfj;

		dev_vel0[i] += dev_qm * dev_ef[0] * dev_dt[i];
		dev_vel1[i] += dev_qm * dev_ef[1] * dev_dt[i];

	}
}
