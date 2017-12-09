/* *****************************************************
 * (c) 2012 Particle In Cell Consulting LLC
 * 
 * This document is subject to the license specified in 
 * Starfish.java and the LICENSE file
 * *****************************************************/

package starfish.sources;

import org.w3c.dom.Element;
import starfish.core.boundaries.Boundary;
import starfish.core.boundaries.Spline;
import starfish.core.common.Starfish;
import starfish.core.common.Vector;
import starfish.core.domain.Mesh;
import starfish.core.io.InputParser;
import starfish.core.materials.KineticMaterial;
import starfish.core.materials.KineticMaterial.Particle;
import starfish.core.materials.Material;
import starfish.core.source.Source;
import starfish.core.source.SourceModule;

/** produces source with cosine distribution used for conductance testing*/
public class CosineSource extends Source
{
    final double den0;
    final double v_drift;
	
    public CosineSource (String name, Material source_mat, Spline spline, 
		    double mdot, double v_drift)
    {
	    super(name,source_mat,spline,mdot);
		
	    /*calculate density*/
	    double A = spline.area();
	    den0 = mdot/(A*v_drift*source_mat.getMass());
	
	    this.v_drift = v_drift;
    }

    @Override
    public Particle sampleParticle()
    {
	Particle part = new Particle((KineticMaterial)source_mat);
	double t = spline.randomT();
	double x[] = spline.pos(t);
	
	/*copy values*/
	part.pos[0] = x[0];
	part.pos[1] = x[1];
	part.pos[2] = 0;
	
	/*velocity*/
	part.vel = Vector.lambertianVector(spline.normal(t), spline.tangent(t));

	/*set velocity and scale by drift*/
	for (int i=0;i<3;i++)
	    part.vel[i] *= v_drift;
	
	num_mp--;
	return part;
    }
   
    @Override
    public void sampleFluid() 
    {
	for (Mesh mesh:Starfish.getMeshList())
	{
	    int ni = mesh.ni;
	    int nj = mesh.nj;
	    double den[][] = source_mat.getDen(mesh).getData();
	    double U[][] = source_mat.getU(mesh).getData();
	    for (int i=0;i<ni;i++)
		for (int j=0;j<nj;j++)
		{
		    den[i][j]=den0;
		    U[i][j]=v_drift;
		}
	}
    }
    
    static public SourceModule.SurfaceSourceFactory cosineSourceFactory = new SourceModule.SurfaceSourceFactory()
    {
	@Override
	public void makeSource(Element element, String name, Boundary boundary, Material material)
	{
	    /*mass flow rate*/
	    double mdot = Double.parseDouble(InputParser.getValue("mdot", element));

	    /*drift velocity*/
	    double v_drift = Double.parseDouble(InputParser.getValue("v_drift", element));

	    CosineSource source = new CosineSource(name, material, boundary, mdot, v_drift);
	    boundary.addSource(source);

	    /*log*/
	    Starfish.Log.log("Added COSINE source '" + name + "'");
	    Starfish.Log.log("> mdot = " + mdot + "(kg/s)");
	    Starfish.Log.log("> flux = " + mdot/boundary.area() + "(kg/s/m^2)");
	    Starfish.Log.log("> spline  = " + boundary.getName());
	    Starfish.Log.log("> v_drift  = " + v_drift);
	}
    };
    
}
