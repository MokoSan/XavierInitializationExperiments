import numpy, pygame, math

pygame.surfarray.use_arraytype('numpy')
#-------------------------------------------------
def buildimage(rval_points, rval_nbpol, nb_poly_max, batchsize, rval_bg,rval_fg, img_shape, neg, **dic):
    surface = pygame.Surface(img_shape, depth=8)
    surface_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surface))
    
    rval_image = numpy.ndarray((batchsize, img_shape[0], img_shape[1]), dtype='uint8')
    rval_image_flat = rval_image.reshape(batchsize, img_shape[0] * img_shape[1])
    
    for j in xrange(batchsize):
        surface.fill(int(int(rval_bg[j])))
    
        for i in range (rval_nbpol[j]):
            pygame.draw.polygon(surface, int(int(rval_fg[j,i])), rval_points[nb_poly_max*j+i], 0)
        rval_image[j] = surface_ndarray
    return rval_image_flat / 255.0 if not neg else (rval_image_flat / 255.0)*2-1
#----------------------------------------------------

def buildsegmentation(rval_points, rval_nbpol, nb_poly_max, batchsize, img_shape, depthmap, neg, neighbor = 'V8', **dic):
    
    if neighbor is 'V4':
        nmult = 2
    else:
        nmult = 4
    
    depth = depthmap.reshape(batchsize,img_shape[0],img_shape[1])
    
    rval_segmentation = numpy.zeros((batchsize, img_shape[0]*nmult, img_shape[1]), dtype='bool')
    rval_segmentation_flat = rval_segmentation.reshape(batchsize, img_shape[0] * img_shape[1] * nmult)
    
    for j in xrange(batchsize):
        maxi = numpy.zeros(2,'uint8')
        mini = numpy.asarray((img_shape[0],img_shape[1]),'uint8')
    
        for i in range(rval_nbpol[j]):
            maxi = numpy.asarray((maxi, rval_points[nb_poly_max*j+i].max(0))).max(0)
            mini = numpy.asarray((mini, rval_points[nb_poly_max*j+i].min(0))).min(0)
        x1 = int(max(0,mini[0]-3))
        x2 = int(min(img_shape[0],maxi[0]+3))
        y1 = int(max(0,mini[1]-3))
        y2 = int(min(img_shape[1],maxi[1]+3))
    
        rval_segmentation[j,x1:x2-1,y1:y2] = (depth[j,x1:x2-1,y1:y2] != depth[j,x1+1:x2,y1:y2]) * 255
        rval_segmentation[j,x1+img_shape[0]:x2+img_shape[0],y1:y2-1] = \
            (depth[j,x1:x2,y1:y2-1] != depth[j,x1:x2,y1+1:y2])*255
        if nmult != 2:
            rval_segmentation[j,x1+2*img_shape[0]:x2+2*img_shape[0]-1,y1:y2-1] =  \
                (depth[j,x1:x2-1,y1:y2-1] !=depth[j,x1+1:x2,y1+1:y2]) * 255
            rval_segmentation[j,x1+3*img_shape[0]+1:x2+3*img_shape[0],y1:y2-1] =  \
                (depth[j,x1+1:x2,y1:y2-1] != depth[j,x1:x2-1,y1+1:y2])*255
    
    return rval_segmentation_flat*1.0 if not neg else ((rval_segmentation_flat)*2-1)*1.0


#--------------------------------------------------------

def buildidentity(rval_points, rval_nbpol, nb_poly_max, rval_poly_id, n_vertices, batchsize, img_shape, neg, **dic):
    surfaceidentity = pygame.Surface(img_shape, depth=8)
    surfaceidentity_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surfaceidentity))
    
    rval_identity = numpy.ndarray((batchsize, len(n_vertices), img_shape[0], img_shape[1]), dtype='bool')
    rval_identity_flat = rval_identity.reshape(batchsize,len(n_vertices) * img_shape[0] * img_shape[1])
    
    for j in xrange(batchsize):
        surfaceidentity.fill(255)
    
        for i in range (rval_nbpol[j]):
            pygame.draw.polygon(surfaceidentity,int(int(rval_poly_id[j,i])), rval_points[nb_poly_max*j+i], 0)
        for i in range(len(n_vertices)):
            rval_identity[j,i] = (surfaceidentity_ndarray == i)
    return rval_identity_flat*1.0 if not neg else ((rval_identity_flat)*2-1)*1.0
#---------------------------------------------------------

def builddepthmap(rval_points, rval_nbpol, nb_poly_max,batchsize,img_shape, neg, **dic):
    surfacedepth = pygame.Surface(img_shape, depth=8)
    surfacedepth_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surfacedepth))
    
    rval_depth = numpy.ndarray((batchsize, img_shape[0], img_shape[1]), dtype='uint8')
    rval_depth_flat = rval_depth.reshape(batchsize, img_shape[0] * img_shape[1])
    
    for j in xrange(batchsize):
        surfacedepth.fill(0)
    
        for i in range (rval_nbpol[j]):
            pygame.draw.polygon(surfacedepth,int(((i+1.0)/(rval_nbpol[j]))*255),rval_points[nb_poly_max*j+i],0)
        rval_depth[j] = surfacedepth_ndarray
    return rval_depth_flat/255.0 if not neg else (rval_depth_flat/255.0)*2-1

# --------------------------------------

def output(rval_poly_id, n_vertices, nb_poly_max,batchsize, **dic):
    rval_output = numpy.zeros((batchsize, len(n_vertices)), dtype='int')
    tmp = numpy.ones(nb_poly_max, dtype='uint8')
    for j in xrange(batchsize):
        for i in range(len(n_vertices)):
            rval_output[j,i] = ((rval_poly_id[j,:] == tmp*i).sum())
    
    return rval_output*1.0

#----------------------------------------------------------

def buildedges(rval_points, rval_nbpol, nb_poly_max, batchsize, img_shape, segmentation, neg, **dic):
    
    if segmentation.size == img_shape[0] * img_shape[1] * 2:
        nmult = 2
    else:
        nmult = 4
    
    seg=segmentation.reshape(batchsize,nmult,img_shape[0],img_shape[1])
    fullseg=numpy.ones((batchsize,2*nmult,img_shape[0],img_shape[1]),dtype='bool')
    fullseg[:,0,:,:]=1-seg[:,0,:,:]
    fullseg[:,1,1:,:]=1-seg[:,0,:-1,:]
    fullseg[:,2,:,:]=1-seg[:,1,:,:]
    fullseg[:,3,:,1:]=1-seg[:,1,:,:-1]
    if nmult == 4:
        fullseg[:,4,:,:]=1-seg[:,2,:,:]
        fullseg[:,5,1:,1:]=1-seg[:,2,:-1,:-1]
        fullseg[:,6,:,:]=1-seg[:,3,:,:]
        fullseg[:,7,:-1,1:]=1-seg[:,3,1:,:-1]
    
    rval_edges = numpy.ones((batchsize, img_shape[0], img_shape[1]), dtype='bool')
    rval_edges_flat = rval_edges.reshape(batchsize, img_shape[0] * img_shape[1])
    
    
    for j in xrange(batchsize):
        maxi = numpy.zeros(2,'uint8')
        mini = numpy.asarray((img_shape[0],img_shape[1]),'uint8')
    
        for i in range(rval_nbpol[j]):
            maxi = numpy.asarray((maxi, rval_points[nb_poly_max*j+i].max(0))).max(0)
            mini = numpy.asarray((mini, rval_points[nb_poly_max*j+i].min(0))).min(0)
        x1 = int(max(0,mini[0]-3))
        x2 = int(min(img_shape[0],maxi[0]+3))
        y1 = int(max(0,mini[1]-3))
        y2 = int(min(img_shape[1],maxi[1]+3))
    
        if nmult == 2:
            rval_edges[j,x1:x2,y1:y2] = ( ( fullseg[j,0,x1:x2,y1:y2] ) * ( fullseg[j,1,x1:x2,y1:y2] )\
                                        * ( fullseg[j,2,x1:x2,y1:y2] ) * ( fullseg[j,3,x1:x2,y1:y2] ))
        else:
            rval_edges[j,x1:x2,y1:y2] = ( ( fullseg[j,0,x1:x2,y1:y2] ) * ( fullseg[j,1,x1:x2,y1:y2] )\
                                        * ( fullseg[j,2,x1:x2,y1:y2] ) * ( fullseg[j,3,x1:x2,y1:y2] )\
                                        * ( fullseg[j,4,x1:x2,y1:y2] ) * ( fullseg[j,5,x1:x2,y1:y2] )\
                                        * ( fullseg[j,6,x1:x2,y1:y2] ) * ( fullseg[j,7,x1:x2,y1:y2] ))
    
    return (1-rval_edges_flat)*1.0 if not neg else ((1-rval_edges_flat)*2-1)*1.0

#----------------------------------------------------------

def buildedgesangle(rval_points, rval_nbpol, nb_poly_max, batchsize, img_shape, segmentation, neg, **dic):
    
    surfaceedges = pygame.Surface(img_shape, depth=8)
    surfaceedges_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surfaceedges))
    
    if segmentation.size == img_shape[0] * img_shape[1] * 2:
        nmult = 2
    else:
        nmult = 4
    
    seg=segmentation.reshape(batchsize,nmult,img_shape[0],img_shape[1])
    fullseg=numpy.ones((batchsize,2*nmult,img_shape[0],img_shape[1]),dtype='bool')
    fullseg[:,0,:,:] = 1-seg[:,0,:,:]
    fullseg[:,1,1:,:] = 1-seg[:,0,:-1,:]
    fullseg[:,2,:,:] = 1-seg[:,1,:,:]
    fullseg[:,3,:,1:] = 1-seg[:,1,:,:-1]
    if nmult == 4:
        fullseg[:,4,:,:] = 1-seg[:,2,:,:]
        fullseg[:,5,1:,1:] = 1-seg[:,2,:-1,:-1]
        fullseg[:,6,:,:] = 1-seg[:,3,:,:]
        fullseg[:,7,:-1,1:] = 1-seg[:,3,1:,:-1]
    
    
    rval_angles = numpy.zeros((batchsize, 4, img_shape[0], img_shape[1]), dtype='bool')
    rval_angles_flat = rval_angles.reshape(batchsize, 4 * img_shape[0] * img_shape[1])
    
    if nmult == 2:
        shifts = numpy.asarray([[1,0],[0,1],[-1,0],[0,-1]])
    if nmult == 4:
        shifts=numpy.asarray([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
    
    for j in xrange(batchsize):
        maxi = numpy.zeros(2,'uint8')
        mini = numpy.asarray((img_shape[0],img_shape[1]),'uint8')
        rval_edges = numpy.ones((img_shape[0], img_shape[1]), dtype='bool')
        surfaceedges.fill(0)
    
        for i in range(rval_nbpol[j]):
            points = rval_points[nb_poly_max*j+i]
            angl = numpy.zeros(len(points),'uint8')
            center = points.mean(0)
            pointsctr = points - center
    
            for k1 in range(len(points)):
                if k1 == len(points)-1:
                    k2 = 0
                else:
                    k2 = k1+1
                edge = points[k2,:] - points[k1,:]
                edge = edge / math.sqrt(pow(edge[0],2)+pow(edge[1],2))
                angltmp = math.atan2(-edge[1],edge[0]) % math.pi #because y is inverted in the image
                if angltmp <= math.pi/8 or angltmp > 7*math.pi/8:
                    angl[k1] = 1
                elif angltmp <= 3*math.pi/8 and angltmp > math.pi/8:
                    angl[k1] = 2
                elif angltmp <= 5*math.pi/8 and angltmp > 3*math.pi/8:
                    angl[k1] = 3
                elif angltmp <= 7*math.pi/8 and angltmp > 5*math.pi/8:
                    angl[k1] = 4
                tmp1 = numpy.dot(shifts,pointsctr[k1,:])+numpy.dot(shifts,pointsctr[k2,:])
                tmp2 = tmp1.argsort()
                for l in tmp2[-nmult-1:]:
                    pygame.draw.polygon(surfaceedges,int(int(angl[k1])),\
                        [center+shifts[l,:],points[k1,:]+shifts[l,:],points[k2,:]+shifts[l,:]],0)
    
            for k1 in range(len(points)):
                if k1 == len(points)-1:
                    k2 = 0
                else:
                    k2 = k1+1
                pygame.draw.polygon(surfaceedges,int(int(angl[k1])),\
                        [(center+points[k2,:]+points[k1,:])/3.0,points[k1,:],points[k2,:]],0)
    
            maxi = numpy.asarray((maxi, rval_points[nb_poly_max*j+i].max(0))).max(0)
            mini = numpy.asarray((mini, rval_points[nb_poly_max*j+i].min(0))).min(0)
    
        x1 = int(max(0,mini[0]-3))
        x2 = int(min(img_shape[0],maxi[0]+3))
        y1 = int(max(0,mini[1]-3))
        y2 = int(min(img_shape[1],maxi[1]+3))
    
        if nmult == 2:
            rval_edges[x1:x2,y1:y2] = ( (fullseg[j,0,x1:x2,y1:y2] ) * (fullseg[j,1,x1:x2,y1:y2] )\
                                        * (fullseg[j,2,x1:x2,y1:y2] ) * (fullseg[j,3,x1:x2,y1:y2] ))
        else:
            rval_edges[x1:x2,y1:y2] = ( (fullseg[j,0,x1:x2,y1:y2] ) * (fullseg[j,1,x1:x2,y1:y2] )\
                                        * (fullseg[j,2,x1:x2,y1:y2] ) * (fullseg[j,3,x1:x2,y1:y2] )\
                                        * (fullseg[j,4,x1:x2,y1:y2] ) * (fullseg[j,5,x1:x2,y1:y2] )\
                                        * (fullseg[j,6,x1:x2,y1:y2] ) * (fullseg[j,7,x1:x2,y1:y2] ))
    
        surfaceedges_ndarray1 = (surfaceedges_ndarray == 1)
        surfaceedges_ndarray2 = (surfaceedges_ndarray == 2)
        surfaceedges_ndarray3 = (surfaceedges_ndarray == 3)
        surfaceedges_ndarray4 = (surfaceedges_ndarray == 4)
        irval_edges = 1-rval_edges
    
        rval_angles[j,0,:,:] =  ((surfaceedges_ndarray1) + (surfaceedges_ndarray2))*(irval_edges)
        rval_angles[j,1,:,:] =  ((surfaceedges_ndarray2) + (surfaceedges_ndarray3))*(irval_edges)
        rval_angles[j,2,:,:] =  ((surfaceedges_ndarray3) + (surfaceedges_ndarray4))*(irval_edges)
        rval_angles[j,3,:,:] =  ((surfaceedges_ndarray4) + (surfaceedges_ndarray1))*(irval_edges)
    
    return rval_angles_flat*1.0 if not neg else ((rval_angles_flat)*2-1)*1.0

#----------------------------------------------------------

class Edgefilter(object):
    def __init__(self,img_shape = (64,64), gaussfiltbool = False, sigma = 0.5, size = 3):
        self.gxprewitt = numpy.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.gyprewitt = numpy.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
        self.gxsobel = numpy.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.gysobel = numpy.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
        self.fftshape =(pow(2,self._nextpow2(img_shape[0]+2*(3+gaussfiltbool*size))),\
                                                            pow(2,self._nextpow2(img_shape[1]+2*(3+gaussfiltbool*size))))
    
        self.fftgxprewitt = numpy.fft.rfft2(self.gxprewitt,self.fftshape)
        self.fftgyprewitt = numpy.fft.rfft2(self.gyprewitt,self.fftshape)
    
        self.fftgxsobel = numpy.fft.rfft2(self.gxsobel,self.fftshape)
        self.fftgysobel = numpy.fft.rfft2(self.gysobel,self.fftshape)
    
        self.sigma = sigma
        self.size = size
        self.gaussfilt = self._gaussian(sigma, size)
    
        self.fftgaussfilt = numpy.fft.rfft2(self.gaussfilt,self.fftshape)
    
    
    def _nextpow2(self,n,i=1):
        if n/2.0 > 1:
            i=self._nextpow2(n/2.0,i+1)
        return i
    
    def _gaussian(self,sigma=0.5, shape=None):
        sigma = max(abs(sigma), 1e-10)
        if shape is None:
            shape = max(int(6*sigma+0.5), 1)
        if not isinstance(shape, tuple):
            shape = (shape, shape)
        x = numpy.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
        y = numpy.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
        Kx = numpy.exp(-x**2/(2*sigma**2))
        Ky = numpy.exp(-y**2/(2*sigma**2))
        ans = numpy.outer(Kx, Ky) / (2.0*numpy.pi*sigma**2)
        return ans/sum(sum(ans))
    
    
    def buildedgesfilter(self,image, batchsize, img_shape, rval_bg, neg, edgetype='sobel', gaussfiltbool = False, \
                    sigma=0.5, size =3, **dic):
        image=image.reshape(batchsize, img_shape[0], img_shape[1])
    
        rval_angles = numpy.zeros((batchsize, 4, img_shape[0], img_shape[1]), dtype='bool')
        rval_angles_flat = rval_angles.reshape(batchsize, 4 * img_shape[0] * img_shape[1])
    
        fftshape =(pow(2,self._nextpow2(img_shape[0]+2*(3+gaussfiltbool*size))),\
                                                            pow(2,self._nextpow2(img_shape[1]+2*(3+gaussfiltbool*size))))
    
        gaussupdate = False
    
        if gaussfiltbool and (self.sigma != sigma or self.size != size):
            self.gaussfilt = self._gaussian(sigma, size)
            self.sigma = sigma
            self.size = size
            self.fftgaussfilt = numpy.fft.rfft2(self.gaussfilt,fftshape)
            gaussupdate = True
    
        if fftshape != self.fftshape:
            self.fftshape = fftshape
            self.fftgxprewitt = numpy.fft.rfft2(self.gxprewitt,self.fftshape)
            self.fftgyprewitt = numpy.fft.rfft2(self.gyprewitt,self.fftshape)
    
            self.fftgxsobel = numpy.fft.rfft2(self.gxsobel,self.fftshape)
            self.fftgysobel = numpy.fft.rfft2(self.gysobel,self.fftshape)
            if not gaussupdate:
                self.fftgaussfilt = numpy.fft.rfft2(self.gaussfilt,self.fftshape)
    
        if edgetype == 'prewitt':
            fftgx = self.fftgxprewitt
            fftgy = self.fftgyprewitt
        else:
            fftgx = self.fftgxsobel
            fftgy = self.fftgysobel
    
        for k in xrange(batchsize):
            image2 = numpy.ones((img_shape[0]+2*(3+gaussfiltbool*size),img_shape[1]+2*(3+gaussfiltbool*size)),\
                        dtype='uint8')*rval_bg[k]
            image2[3+gaussfiltbool*size:-3-gaussfiltbool*size,3+gaussfiltbool*size:-3-gaussfiltbool*size]=image[ k,:,:]
            fftimage = numpy.fft.rfft2(image2[:,:],fftshape)
            if gaussfiltbool:
                igx = numpy.fft.irfft2(fftimage*self.fftgaussfilt*fftgx,fftshape)
                igy = numpy.fft.irfft2(fftimage*self.fftgaussfilt*fftgy,fftshape)
            else:
                igx = numpy.fft.irfft2(fftimage*fftgx,fftshape)
                igy = numpy.fft.irfft2(fftimage*fftgy,fftshape)
            igx = igx[4+gaussfiltbool*(size+size/2):img_shape[0]+4+gaussfiltbool*(size+size/2),\
                            4+gaussfiltbool*(size+size/2):img_shape[1]+4+gaussfiltbool*(size+size/2)]
            igy = igy[4+gaussfiltbool*(size+size/2):img_shape[0]+4+gaussfiltbool*(size+size/2),\
                            4+gaussfiltbool*(size+size/2):img_shape[1]+4+gaussfiltbool*(size+size/2)]
            edgesdetec = (numpy.sqrt(igx*igx+igy*igy) >= 1.0)
            angl = numpy.arctan2(igy,igx) % math.pi
            tmp=numpy.ones(angl.shape)
    
            angl1 = ((angl<=tmp*math.pi/8) + (angl>tmp*7*math.pi/8))
            angl2 = ((angl<=tmp*3*math.pi/8) * (angl>tmp*math.pi/8))
            angl3 = ((angl<=tmp*5*math.pi/8) * (angl>tmp*3*math.pi/8))
            angl4 = ((angl<=tmp*7*math.pi/8) * (angl>tmp*5*math.pi/8))
    
            rval_angles[k,0,:,:] =  ((angl1) + (angl2))*edgesdetec
            rval_angles[k,1,:,:] =  ((angl2) + (angl3))*edgesdetec
            rval_angles[k,2,:,:] =  ((angl3) + (angl4))*edgesdetec
            rval_angles[k,3,:,:] =  ((angl4) + (angl1))*edgesdetec
        return rval_angles_flat if not neg else (rval_angles_flat)*2-1
    
    
    def buildedgesfilterc(self,image, batchsize, img_shape, rval_bg, neg, edgetype='sobel', gaussfiltbool = False,\
                                sigma = 0.5, size=3, **dic):
        image=image.reshape(batchsize, img_shape[0], img_shape[1])
    
        rval_angles = numpy.zeros((batchsize, 4, img_shape[0], img_shape[1]), dtype='double')
        rval_angles_flat = rval_angles.reshape(batchsize, 4 * img_shape[0] * img_shape[1])
    
        fftshape =(pow(2,self._nextpow2(img_shape[0]+2*(3+gaussfiltbool*size))),\
                                                            pow(2,self._nextpow2(img_shape[1]+2*(3+gaussfiltbool*size))))
    
        gaussupdate = False
    
        if gaussfiltbool and (self.sigma != sigma or self.size != size):
            self.gaussfilt = self._gaussian(sigma, size)
            self.sigma = sigma
            self.size = size
            self.fftgaussfilt = numpy.fft.rfft2(self.gaussfilt,fftshape)
            gaussupdate = True
    
        if fftshape != self.fftshape:
            self.fftshape = fftshape
            self.fftgxprewitt = numpy.fft.rfft2(self.gxprewitt,self.fftshape)
            self.fftgyprewitt = numpy.fft.rfft2(self.gyprewitt,self.fftshape)
    
            self.fftgxsobel = numpy.fft.rfft2(self.gxsobel,self.fftshape)
            self.fftgysobel = numpy.fft.rfft2(self.gysobel,self.fftshape)
            if not gaussupdate:
                self.fftgaussfilt = numpy.fft.rfft2(self.gaussfilt,self.fftshape)
    
        if edgetype == 'prewitt':
            fftgx = self.fftgxprewitt
            fftgy = self.fftgyprewitt
        else:
            fftgx = self.fftgxsobel
            fftgy = self.fftgysobel
    
        for k in xrange(batchsize):
            image2 = numpy.ones((img_shape[0]+2*(3+gaussfiltbool*size),img_shape[1]+2*(3+gaussfiltbool*size)),\
                        dtype='uint8')*rval_bg[k]
            image2[3+gaussfiltbool*size:-3-gaussfiltbool*size,3+gaussfiltbool*size:-3-gaussfiltbool*size]=image[ k,:,:]
            fftimage = numpy.fft.rfft2(image2[:,:],fftshape)
            if gaussfiltbool:
                igx = numpy.fft.irfft2(fftimage*self.fftgaussfilt*fftgx,fftshape)
                igy = numpy.fft.irfft2(fftimage*self.fftgaussfilt*fftgy,fftshape)
            else:
                igx = numpy.fft.irfft2(fftimage*fftgx,fftshape)
                igy = numpy.fft.irfft2(fftimage*fftgy,fftshape)
            igx = igx[4+gaussfiltbool*(size+size/2):img_shape[0]+4+gaussfiltbool*(size+size/2),\
                            4+gaussfiltbool*(size+size/2):img_shape[1]+4+gaussfiltbool*(size+size/2)]
            igy = igy[4+gaussfiltbool*(size+size/2):img_shape[0]+4+gaussfiltbool*(size+size/2),\
                            4+gaussfiltbool*(size+size/2):img_shape[1]+4+gaussfiltbool*(size+size/2)]
    
            edgesdetec = (numpy.sqrt(igx*igx+igy*igy) >= 1.0)
    
            angl = numpy.arctan2(igy,igx) % math.pi
    
            tmp=numpy.ones(angl.shape)
    
            angl1 = (angl<tmp*math.pi/4)
            angl2 = ((angl<tmp*math.pi/2) * (angl>=tmp*math.pi/4))
            angl3 = ((angl<tmp*3*math.pi/4) * (angl>=tmp*math.pi/2))
            angl4 = ((angl<tmp*math.pi) * (angl>=tmp*3*math.pi/4))
    
            angl=(angl % (math.pi/4)) / (math.pi/4)
            iangl=1-angl
    
            rval_angles[k,0,:,:] =  ((angl1) * (iangl) +\
                                     (angl4) * angl)*edgesdetec
            rval_angles[k,1,:,:] =  ((angl2) * (iangl) +\
                                     (angl1) * angl)*edgesdetec
            rval_angles[k,2,:,:] =  ((angl3) * (iangl) +\
                                     (angl2) * angl)*edgesdetec
            rval_angles[k,3,:,:] =  ((angl4) * (iangl) +\
                                     (angl3) * angl)*edgesdetec
        return rval_angles_flat if not neg else (rval_angles_flat)*2-1

edgefilt = Edgefilter()
buildedgesfilter = edgefilt.buildedgesfilter
buildedgesfilterc = edgefilt.buildedgesfilterc


#----------------------------------------------------------

#other features versions:
#--------------------------------------------------------

def buildidentityc(rval_points, rval_nbpol, nb_poly_max, rval_poly_id, n_vertices, batchsize, img_shape, neg, **dic):
    surfaceidentity = pygame.Surface(img_shape, depth=8)
    surfaceidentity_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surfaceidentity))
    
    rval_identity = numpy.ndarray((batchsize, img_shape[0], img_shape[1]), dtype='uint8')
    rval_identity_flat = rval_identity.reshape(batchsize, img_shape[0] * img_shape[1])
    
    for j in xrange(batchsize):
        surfaceidentity.fill(0)
    
        for i in range (rval_nbpol[j]):
            pygame.draw.polygon(surfaceidentity,int(((rval_poly_id[j,i]+1.0)/(len(n_vertices)))*255),\
                rval_points[nb_poly_max*j+i], 0)
        rval_identity[j] = surfaceidentity_ndarray
    return rval_identity_flat / 255.0 if not neg else (rval_identity_flat / 255.0)*2-1

#----------------------------------------------------------

def buildedgesanglec(rval_points, rval_nbpol, nb_poly_max, batchsize, img_shape, segmentation, neg, **dic):
    
    surfaceedges = pygame.Surface(img_shape, depth=8)
    surfaceedges_ndarray = numpy.asarray(pygame.surfarray.pixels2d(surfaceedges))
    
    if segmentation.size == img_shape[0] * img_shape[1] * 2:
        nmult = 2
    else:
        nmult = 4
    
    seg=segmentation.reshape(batchsize,nmult,img_shape[0],img_shape[1])
    fullseg=numpy.ones((batchsize,2*nmult,img_shape[0],img_shape[1]),dtype='bool')
    fullseg[:,0,:,:] = 1-seg[:,0,:,:]
    fullseg[:,1,1:,:] = 1-seg[:,0,:-1,:]
    fullseg[:,2,:,:] = 1-seg[:,1,:,:]
    fullseg[:,3,:,1:] = 1-seg[:,1,:,:-1]
    if nmult == 4:
        fullseg[:,4,:,:] = 1-seg[:,2,:,:]
        fullseg[:,5,1:,1:] = 1-seg[:,2,:-1,:-1]
        fullseg[:,6,:,:] = 1-seg[:,3,:,:]
        fullseg[:,7,:-1,1:] = 1-seg[:,3,1:,:-1]
    
    
    rval_angles = numpy.zeros((batchsize, 4, img_shape[0], img_shape[1]), dtype='double')
    rval_angles_flat = rval_angles.reshape(batchsize, 4 * img_shape[0] * img_shape[1])
    
    if nmult == 2:
        shifts = numpy.asarray([[1,0],[0,1],[-1,0],[0,-1]])
    if nmult == 4:
        shifts=numpy.asarray([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
    
    for j in xrange(batchsize):
        maxi = numpy.zeros(2,'uint8')
        mini = numpy.asarray((img_shape[0],img_shape[1]),'uint8')
        rval_edges = numpy.ones((img_shape[0], img_shape[1]), dtype='bool')
        surfaceedges.fill(0)
    
        for i in range(rval_nbpol[j]):
            points = rval_points[nb_poly_max*j+i]
            angltmp = numpy.zeros(len(points),'uint8')
            center = points.mean(0)
            pointsctr = points - center
    
            for k1 in range(len(points)):
                if k1 == len(points)-1:
                    k2 = 0
                else:
                    k2 = k1+1
                edge = points[k2,:] - points[k1,:]
                edge = edge[:] / math.sqrt(pow(edge[0],2)+pow(edge[1],2))
                angltmp[k1] = round((math.atan2(-edge[1],edge[0]) % math.pi) / math.pi * 255)
                #because y is inverted in the image drawing
                tmp1 = numpy.dot(shifts,pointsctr[k1,:])+numpy.dot(shifts,pointsctr[k2,:])
                tmp2 = tmp1.argsort()
                for l in tmp2[-nmult-1:]:
                    pygame.draw.polygon(surfaceedges,int(int(angltmp[k1])),\
                        [center+shifts[l,:],points[k1,:]+shifts[l,:],points[k2,:]+shifts[l,:]],0)
    
            for k1 in range(len(points)):
                if k1 == len(points)-1:
                    k2 = 0
                else:
                    k2 = k1+1
                pygame.draw.polygon(surfaceedges,int(int(angltmp[k1])),\
                        [(center+points[k2,:]+points[k1,:])/3.0,points[k1,:],points[k2,:]],0)
            
            maxi = numpy.asarray((maxi, rval_points[nb_poly_max*j+i].max(0))).max(0)
            mini = numpy.asarray((mini, rval_points[nb_poly_max*j+i].min(0))).min(0)
    
        x1 = int(max(0,mini[0]-3))
        x2 = int(min(img_shape[0],maxi[0]+3))
        y1 = int(max(0,mini[1]-3))
        y2 = int(min(img_shape[1],maxi[1]+3))
    
        if nmult == 2:
            rval_edges[x1:x2,y1:y2] = ( (fullseg[j,0,x1:x2,y1:y2] ) * (fullseg[j,1,x1:x2,y1:y2] )\
                                        * (fullseg[j,2,x1:x2,y1:y2] ) * (fullseg[j,3,x1:x2,y1:y2] ))
        else:
            rval_edges[x1:x2,y1:y2] = ( (fullseg[j,0,x1:x2,y1:y2] ) * (fullseg[j,1,x1:x2,y1:y2] )\
                                        * (fullseg[j,2,x1:x2,y1:y2] ) * (fullseg[j,3,x1:x2,y1:y2] )\
                                        * (fullseg[j,4,x1:x2,y1:y2] ) * (fullseg[j,5,x1:x2,y1:y2] )\
                                        * (fullseg[j,6,x1:x2,y1:y2] ) * (fullseg[j,7,x1:x2,y1:y2] ))
    
        tmp=numpy.ones(surfaceedges_ndarray.shape)*255
    
        angl1 = (surfaceedges_ndarray<tmp/4.0)
        angl2 = ((surfaceedges_ndarray<tmp/2.0) * (surfaceedges_ndarray>=tmp/4.0))
        angl3 = ((surfaceedges_ndarray<tmp*3/4.0) * (surfaceedges_ndarray>=tmp/2.0))
        angl4 = ((surfaceedges_ndarray<tmp) * (surfaceedges_ndarray>=tmp*3/4.0))
    
    
        surfaceedgesc = (surfaceedges_ndarray % (255/4.0)) / (255/4.0)
        isurfaceedgesc = 1 - surfaceedgesc
        irval_edges = (1-rval_edges)
    
    
        rval_angles[j,0,:,:] =  ((angl1) * (isurfaceedgesc) +\
                                 (angl4) * surfaceedgesc ) * (irval_edges)
        rval_angles[j,1,:,:] =  ((angl2) * (isurfaceedgesc) +\
                                 (angl1) * surfaceedgesc ) * (irval_edges)
        rval_angles[j,2,:,:] =  ((angl3) * (isurfaceedgesc) +\
                                 (angl2) * surfaceedgesc ) * (irval_edges)
        rval_angles[j,3,:,:] =  ((angl4) * (isurfaceedgesc) +\
                                 (angl3) * surfaceedgesc ) * (irval_edges)
    return rval_angles_flat*1.0 if not neg else ((rval_angles_flat)*2-1)*1.0



#def drawpolygon(oldimage,color,points):
    #points = numpy.concatenate([points,numpy.asarray([points[0,:]])],0)
    #mini = points.min(0)
    #maxi = points.max(0)
    #idx_x = [i for i in xrange(int(numpy.floor(mini[0])),int(numpy.ceil(maxi[0])),1)]
    #idx_y = [i for i in xrange(int(numpy.floor(mini[1])),int(numpy.ceil(maxi[1])),1)]
    #A = numpy.ndarray((4,len(idx_x)*len(idx_y),points.shape[0],points.shape[1]),dtype = 'float64')
    #for i in range(len(idx_x)):
        #for j in range(len(idx_y)):
            #A[0,i*len(idx_y)+j,:,:] = points-\
                    #numpy.transpose(numpy.asarray([[idx_x[i]]*points.shape[0],[idx_y[j]]*points.shape[0]]))
            #A[1,i*len(idx_y)+j,:,:] = points-\
                    #numpy.transpose(numpy.asarray([[idx_x[i]+1]*points.shape[0],[idx_y[j]]*points.shape[0]]))
            #A[2,i*len(idx_y)+j,:,:] = points-\
                    #numpy.transpose(numpy.asarray([[idx_x[i]+1]*points.shape[0],[idx_y[j]+1]*points.shape[0]]))
            #A[3,i*len(idx_y)+j,:,:] = points-\
                    #numpy.transpose(numpy.asarray([[idx_x[i]]*points.shape[0],[idx_y[j]+1]*points.shape[0]]))
    #B=A[:,:,1:,0]*A[:,:,:-1,1] - A[:,:,:-1,0]*A[:,:,1:,1]
    #b1 = (B > 0)
    #b2 = (B < 0)
    #c1 = b1.prod(2)
    #c2 = b2.prod(2)
    #d = (c1 + c2) != 0
    #f = (d[0,:] + d[1,:] + d[2,:] + d[3,:]) != 0
    #e = numpy.reshape(f,(len(idx_x),len(idx_y)))
    #print e
    #oldimage[idx_x[0]:idx_x[-1]+1,idx_y[0]:idx_y[-1]+1] = \
            #oldimage[idx_x[0]:idx_x[-1]+1,idx_y[0]:idx_y[-1]+1]*(e==0)+(e==1)*color
    #return oldimage
