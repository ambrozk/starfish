/* *****************************************************
 * (c) 2012 Particle In Cell Consulting LLC
 * 
 * This document is subject to the license specified in 
 * Starfish.java and the LICENSE file
 * *****************************************************/

package starfish.core.domain;

/**
 * FieldCollection collects fields corresponding to the same physical quantity 
 * (number density, etc...) across multiple meshes. It can be created in one of three ways:
 * 1) FieldCollection(Mesh): allocates space for the variable on the specified mesh
 * 2) FieldCollection(ArrayList<mesh>): like 1) but for the specified mesh list
 * 3) FieldCollection(Field): wraps an existing field into a single mesh field collection
 */

import java.util.HashMap;
import java.util.NoSuchElementException;
import java.util.Set;
import starfish.core.common.Starfish;
import starfish.core.domain.Mesh.BoundaryData;
import starfish.core.domain.Mesh.Face;
import starfish.core.domain.Mesh.NodeType;

/**inner class holding fields for a particular mesh*/
public class FieldCollection2D
{
    public FieldCollection2D(Iterable<Mesh> mesh_list, MeshEvalFun eval_fun)
    {
	/*add fields for each mesh*/
	for (Mesh mesh:mesh_list)
	{
	    fields.put(mesh, new Field2D(mesh));
	}
	
	this.eval_fun = eval_fun;
    }

    /*adds field collection to just a single mesh*/
    public FieldCollection2D(Mesh mesh, MeshEvalFun eval_fun)
    {
	fields.put(mesh, new Field2D(mesh));
	this.eval_fun = eval_fun;
    }

    public FieldCollection2D(Field2D field, MeshEvalFun eval_fun)
    {
	fields.put(field.getMesh(), field);
	this.eval_fun = eval_fun;
    }

    public FieldCollection2D(FieldCollection2D fc) 
    {
	this(fc.getMeshes(),fc.getEvalFun());
    }

    public void copy(FieldCollection2D fc)
    {
	for (Mesh mesh: Starfish.getMeshList())
	    getField(mesh).copy(fc.getField(mesh));
    }

    public void mult(double val)
    {
	for (Mesh mesh: Starfish.getMeshList())
	    getField(mesh).mult(val);
    }

    public int it_last_eval = -1;
    
    /**uses this field's evalFun to update field values*/
    public void eval()
    {
	if (Starfish.getIt()>it_last_eval && eval_fun!=null)
	{
	    eval_fun.eval(this);
	    it_last_eval = Starfish.getIt();
	}
    }
    
    final protected MeshEvalFun eval_fun;
    
    /**eval fun interface*/
    public interface MeshEvalFun {
	public void eval(FieldCollection2D fc2);
    }
    public MeshEvalFun getEvalFun() {return eval_fun;}
    
    /**returns field for the mesh, creates a new field if not yet in the collection*/
    public Field2D getField(Mesh mesh) 
    {
	/*do we already have this mesh in the list?*/
	Field2D field = fields.get(mesh);
	if (field==null)
	{
	    field = new Field2D(mesh);
	    fields.put(mesh, field);
	}
	return field;
    }
	
    public Field2D[] getFields() {return fields.values().toArray(new Field2D[fields.values().size()]);}
    public Set<Mesh> getMeshes() {return fields.keySet();}
	
    HashMap<Mesh,Field2D> fields = new HashMap<Mesh,Field2D>();
		
    /**creates continuity across mesh boundaries*/
    public void syncMeshBoundaries()
    {
	/*pass 1: add values to buffer*/
	for (Mesh mesh:Starfish.getMeshList())
	{
	    for (Face face:Face.values())
	    {
		BoundaryData bc[] = mesh.boundary_data[face.ordinal()];
		int i,j;
		
		if (face==Face.BOTTOM || face==Face.TOP)
		{
		    int j2;
		    if (face==Face.BOTTOM) {j=0;j2=mesh.nj-1;}
		    else {j=mesh.nj-1;j2=0;}
					
		    for (i=0;i<mesh.ni;i++)
		    {
			if (mesh.node[i][j].type == NodeType.MESH)
			{
			    /*TODO: hardcoded for a single neighbor*/
			    Mesh nm = bc[i].neighbor[0];
			    double x[] = mesh.pos(i,j);
			    if (nm.containsPos(x))
			    {
				double lc[]=nm.XtoL(x);
				bc[i].buffer+=getField(nm).gather_safe(lc);
			    }							
			}
			else if (mesh.node[i][j].type == NodeType.PERIODIC)
			{
			    /*TODO: hardcoded for single cartesian mesh!*/
			    bc[i].buffer+=getField(mesh).data[i][j2];
			}
		    }
		}
		else if (face==Face.LEFT || face==Face.RIGHT)
		{
		    int i2;
		    if (face==Face.LEFT) {i=0; i2=mesh.ni-1;}
		    else {i=mesh.ni-1; i2=0;}			
					
		    for (j=0;j<mesh.nj;j++)
		    {
			if (mesh.node[i][j].type == NodeType.MESH)
			{
			    /*TODO: hardcoded for a single neighbor*/
			    Mesh nm = bc[j].neighbor[0];
			    double x[] = mesh.pos(i,j);
			    if (nm.containsPos(x))
			    {
				double lc[]=nm.XtoL(x);
				bc[j].buffer+=getField(nm).gather_safe(lc);
			    }				
			} /*if mesh*/
			else if (mesh.node[i][j].type == NodeType.PERIODIC)
			{
			    /*TODO: hardcoded for single cartesian mesh!*/
			    bc[j].buffer+=getField(mesh).data[i2][j];
			}
		    } /*j*/
		} /*left or right*/
	    }
	}	
	
	/*pass 2: add buffer to field data*/
	for (Mesh mesh:Starfish.getMeshList())
	{
	    Field2D field = getField(mesh);

	    for (Face face:Face.values())
	    {
		BoundaryData bc[] = mesh.boundary_data[face.ordinal()];
		int i,j;

		if (face==Face.BOTTOM || face==Face.TOP)
		{
		    if (face==Face.BOTTOM) j=0;
		    else j=mesh.nj-1;

		    for (i=0;i<mesh.ni;i++)
		    {
			if (mesh.node[i][j].type == NodeType.MESH ||
			    mesh.node[i][j].type == NodeType.PERIODIC)
			{
			    field.data[i][j]+=bc[i].buffer;
			    field.data1d[i*field.nj+j]+=bc[i].buffer;
			    
			    bc[i].buffer=0;
			}
		    }
		} /*bottom or top*/
		else if (face==Face.LEFT || face==Face.RIGHT)
		{
		    if (face==Face.LEFT) i=0;
		    else i=mesh.ni-1;

		    for (j=0;j<mesh.nj;j++)
		    {
			if (mesh.node[i][j].type == NodeType.MESH ||
			    mesh.node[i][j].type == NodeType.PERIODIC)
			{
			    field.data[i][j]+=bc[j].buffer;
				field.data1d[i*field.nj+j]+=bc[j].buffer;
			    field.data[i][j]*=0.5;
				field.data1d[i*field.nj+j]*=0.5;
			    bc[j].buffer=0;
			}
		    }
		} /*left or right*/
	    }
	}	/*for mesh loop*/	
    }

    /**sets the entire collection to a constant value*/
    public void setValue(double value) 
    {
	for (Field2D field:this.fields.values())
	    field.setValue(value);
    }

    /**adds values from another field collection to this one*/
    public void addData(FieldCollection2D other)
    {
	for (Mesh mesh:getMeshes())
	{
	    this.getField(mesh).add(other.getField(mesh));
	}
    }
    
    public double[] getRange() 
    {
	double range[] = new double[2];
	range[0] = 1e66;
	range[1] = -1e66;

	for (Field2D field:fields.values())
	{
	    double fr[] = field.getRange();
	    if (fr[0]<range[0]) range[0]=fr[0];
	    if (fr[1]>range[1]) range[1]=fr[1];
	}

	return range;
    }

    /**returns value at position x, or throws an exception if
     * the position is not contained within any of the meshes*/
    public double eval(double[] x) 
    {
	for (Mesh mesh:fields.keySet())
	{
	    if (mesh.containsPos(x))
		return fields.get(mesh).eval(x);
	}
	throw new NoSuchElementException(String.format("position (%g, %g) not found",x[0],x[1]));
    }

    /** sets all fields to zero*/
    public void clear() 
    {
	for (Field2D field:fields.values())
	    field.clear();
    }
    
    /*multiplies two field collections by each other and a scalar, and return the product*/
    public static FieldCollection2D mult(FieldCollection2D fc1, FieldCollection2D fc2, double scalar)
    {
	FieldCollection2D res = new FieldCollection2D(fc1.getMeshes(),null);
	fc1.eval();
	fc2.eval();
	for (Mesh mesh:res.getMeshes())
	{
	    double r[][] = res.getField(mesh).getData();
	    double f1[][] = fc1.getField(mesh).getData();
	    double f2[][] = fc2.getField(mesh).getData();
	    
	    for (int i=0;i<mesh.ni;i++)
		for (int j=0;j<mesh.nj;j++)
		{
		    r[i][j] = f1[i][j]*f2[i][j]*scalar;
		}
	}
	return res;	
    }
}
