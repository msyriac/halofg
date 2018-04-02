import healpy as hp
import halofg.sehgalInterface as si
import orphics.tools.catalogs as cats
import orphics.io as io
import numpy as np
import astropy.coordinates
import sys


# patches: y0,x0,y1,x1 (y=DEC deg, x=RA deg)
deep56_box =  [-7.5,38.5,4.,-5.5]
y0,x0,y1,x1 = deep56_box
hwidth = x0-x1
vwidth = y1-y0

def healpix_box_indices(nside,y0,x0,y1,x1):

    ra_left = x0
    ra_right = x1
    dec_down = y0
    dec_up = y1
    
    dec_poly, ra_poly = (np.array([ dec_down,  dec_down,  dec_up,  dec_up]), np.array([ ra_left,  ra_right,  ra_right,  ra_left]))
    xyzpoly = astropy.coordinates.spherical_to_cartesian(1, np.deg2rad(dec_poly), np.deg2rad(ra_poly))
    return hp.query_polygon(nside,np.array(xyzpoly).T)

nside = 128
pix = healpix_box_indices(nside,*deep56_box)
tmap = np.zeros(hp.nside2npix(nside))
tmap[pix] = 1
# io.mollview(tmap,io.dout_dir+"tmap.png")#,coord=['G','C'])
# sys.exit()

SimConfig = io.config_from_file("../halofg/input/sehgal.ini")
PathConfig = io.config_from_file("../halofg/input/paths_local.ini")



df,ra,dec,m200_sehgal,z_sehgal = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_m200mean',M200_min=0.,M200_max=np.inf,z_min=0,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,mirror=False)

hcat = cats.HealpixCatMapper(nside,ra,dec)

frames = 200
# phi = 0.
# thetas = np.linspace(0,30,frames)
# psi = 0.


hmap = hcat.counts
hmap[hmap<1] = 0
hmap[hmap>1] = 1

target = tmap.sum()

io.mollview(tmap,io.dout_dir+"tmap.png")
io.mollview(hmap,io.dout_dir+"hmap.png")


print "Rotating..."



import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# animation function
eulers = []
def animate(i):
    #theta = np.asarray(thetas[i])
    test = 0
    
    while test<0.98:
        phi = np.random.uniform(0.,360.)
        theta = np.random.uniform(0.,360.)
        psi = np.random.uniform(0.,360.)
    
        alm=hp.map2alm(hmap)
        hp.rotate_alm(alm,psi*np.pi/180.,theta*np.pi/180.,phi*np.pi/180.)
        hmap_new=hp.alm2map(alm,nside=nside,verbose=False)

        
        test = np.sum(hmap_new*tmap)/target
        
        print test
        print theta,phi,psi


    print "Passed"
    eulers.append((phi,theta,psi))
    pltimage = io.mollview(tmap*2+hmap_new,return_projected_map=True)
    cont = ax.imshow(np.flipud(pltimage))
    return cont  

anim = animation.FuncAnimation(fig, animate, frames=np.arange(0,frames),interval=400)
 
anim.save(io.dout_dir+'hcat.gif', dpi=80, writer='imagemagick')
# io.save_cols("eulers_98percent.txt",eulers)



# psi1, theta1, phi1 = hp.rotator.coordsys2euler_zyz(['C','G'])
# print psi1,theta1,phi1
# hp.rotate_alm(alm,psi1,theta1,phi1)
# psi2, theta2, phi2 = hp.rotator.coordsys2euler_zyz(['G','C'])
# hp.rotate_alm(alm,psi2,theta2,phi2)
