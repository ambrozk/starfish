
extern "C"
__global__ void pushParticle(Particle *particles, double *ef, double qm, long N)
{
	/*get particle id*/
	long p = blockIdx.x*blockDim.x+threadIdx.x;

	if (p<N && particles[p].alive)
	{
		/*grab pointer to this particle*/
		Particle *part = &particles[p];

		/*compute particle node position*/
		double lc = XtoL(part->x);

		/*gather electric field onto particle position*/
		double part_ef = gather(lc,ef);

		/*advance velocity*/
		part->v += DT*qm*part_ef;

		/*advance position*/
		part->x += DT*part->v;

		/*remove particles leaving the domain*/
		if (part->x < X0 || part->x >= XMAX)
			part->alive = false;
	}
}
