/* *****************************************************
 * (c) 2012 Particle In Cell Consulting LLC
 * 
 * This document is subject to the license specified in 
 * Starfish.java and the LICENSE file
 * *****************************************************/

package starfish.core.domain;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import starfish.core.domain.FieldCollection2D.MeshEvalFun;

/** 
 * Field manager holds a list of FieldCollections
 * 
 * Each FieldCollection collects fields corresponding to a single variable
 * over a number of meshes */
public class FieldManager2D 
{
    /*variable names*/ 
    protected ArrayList<String> names;
    public ArrayList<String> getNames() {return names;}
	
    /*units*/
    protected HashMap<String,String> units;
	
    /*list of meshes*/
    protected ArrayList<Mesh> mesh_list;
    public ArrayList<Mesh> getMeshList() {return mesh_list;}

    /*collection of the named field on all meshes*/
    HashMap<String,FieldCollection2D> field_collection2d;
    public Collection<FieldCollection2D> getFieldCollections() {return field_collection2d.values();}
	
    /**returns true if the field exists*/
    public boolean hasField(String name) { return names.contains(name);}

    /**constructor for a single mesh*/
    public FieldManager2D(Mesh mesh) 
    {
	ArrayList<Mesh> list = new ArrayList<Mesh>();
	list.add(mesh);
    	
	construct(list);
    }
				
    /**constructor for a collection of meshes*/
    public FieldManager2D(ArrayList<Mesh> mesh_list) 
    {
	construct(mesh_list);
    }
	
    /**initialization function*/
    protected final void construct(ArrayList<Mesh> mesh_list) 
    {
	this.mesh_list = mesh_list;		/*save parent*/
		
	/*allocate memory structues*/
	names = new ArrayList<String>();
	units = new HashMap<String,String>();
	field_collection2d = new HashMap<String,FieldCollection2D>();
    }
	
    /**returns field "name" from "mesh"*/
    public Field2D get(Mesh mesh, String name) 
    {
	return field_collection2d.get(name).getField(mesh);	
    }
	
    public FieldCollection2D getFieldCollection(String name) {return field_collection2d.get(name.toLowerCase());}
	
    /**returns units*/
    public String getUnits(String name) {return units.get(name);}

    /**add a new named field or returns the existing one if we already have this*/
    public FieldCollection2D add(String name, String unit, MeshEvalFun eval_fun) 
    {
	/*set to lowercase*/
	name = name.toLowerCase();
		
	FieldCollection2D collection;
		
	/*check if we already have this variable*/
	collection = field_collection2d.get(name);
	if (collection!=null) return collection;
		
	/*save this variable*/
	names.add(name);
	units.put(name,unit);
	
	/*new field collection*/
	collection = new FieldCollection2D(mesh_list, eval_fun);
		
	/*add to list*/
	field_collection2d.put(name, collection);
    	
	return collection;
    }
	
    /**add a new named field and initializes it according to data from Field*/
    public void add(String name, String unit, Field2D values, MeshEvalFun eval_fun) 
    {
	FieldCollection2D collection = add(name,unit,eval_fun);
    	
	Field2D field = collection.getField(values.getMesh());
	Field2D.DataCopy(values.data,field.data);	
    }
	
    /**add a new named field and initializes it to a constant value*/
    public void add(String name, String unit, double value, MeshEvalFun eval_fun) 
    {
	FieldCollection2D collection = add(name,unit,eval_fun);
	collection.setValue(value);
    }
}
