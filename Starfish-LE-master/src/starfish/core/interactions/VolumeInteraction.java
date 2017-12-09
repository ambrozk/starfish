/* *****************************************************
 * (c) 2012 Particle In Cell Consulting LLC
 * 
 * This document is subject to the license specified in 
 * Starfish.java and the LICENSE file
 * *****************************************************/

package starfish.core.interactions;

import starfish.core.domain.Field2D;
import starfish.core.domain.FieldCollection2D;
import starfish.core.domain.Mesh;

/** base class for volume interactions*/
abstract public class VolumeInteraction 
{
    public FieldCollection2D rate;
    public FieldCollection2D dn_source[];
	
    public FieldCollection2D getRate() {return rate;}
    public Field2D getRate(Mesh mesh) {return rate.getField(mesh);}
			
    public FieldCollection2D[] getDnSource() {return dn_source;}
    public Field2D[] getDnSource(Mesh mesh) 
    {
	Field2D f[] = new Field2D[dn_source.length];
	for (int s=0;s<dn_source.length;s++)
	    f[s] = dn_source[s].getField(mesh);
	return f;
    }
	
    abstract public void perform();
    abstract public void init();	
}
