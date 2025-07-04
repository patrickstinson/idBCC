import numpy as np
from scipy.special import softmax, logsumexp


def binary_search(df,interval,bounds,max_reps = 200):
    #assuming function is monotonically increasing on interval and bounds are contained in it

    x = interval.mean()
    reps = 0
    while not (df(x)>=bounds.min() and df(x)<=bounds.max()):
        reps +=1
        if reps>max_reps:
            return None
        if df(x)<bounds.min():
            interval = np.array([x,interval[1]])
        else:
            interval = np.array([interval[0],x])

        x = interval.mean()
    return x
    
    
def ars_nonneg(ff,df,x0,lbounds=np.array([.1,1.]),rbounds=np.array([.1,1.]),nsamps=1):
    
    def find_left_point(dlogp,x,bounds,delta_x):
    
        #too far to left
        if dlogp(x)>bounds.max():
            box = np.array([x,np.nan])

            while dlogp(x)>bounds.max():
                x *= delta_x

            box[1] = x

            dx = binary_search(lambda xx: dlogp(box[1]-xx),
                                    np.array([0,box[1]-box[0]]),bounds)
            left_point = box[1]-dx

            return left_point

        #too far to right
        elif dlogp(x)<bounds.min():
            box = np.array([np.nan,x])

            while dlogp(x)<bounds.min():
                x /= delta_x

            box[0] = x

            dx = binary_search(lambda xx: dlogp(box[1]-xx),
                                    np.array([0,box[1]-box[0]]),bounds)
            left_point = box[1]-dx

            return left_point

        else:
            return x

    def find_right_point(dlogp,x,bounds,delta_x):

        #too far to right
        if dlogp(x)<-bounds.max():
            box = np.array([np.nan,x])

            while dlogp(x)<-bounds.max():
                x /= delta_x

            box[0] = x

            dx = binary_search(lambda x: -dlogp(box[0]+x),
                                np.array([0,box[1]-box[0]]),bounds)
            log_right_point = box[0]+dx

            return log_right_point

        #too far to left
        elif dlogp(x)>-bounds.min():
            box = np.array([x,np.nan])

            while dlogp(x)>-bounds.min():
                #move it a bit farther
                #while dlogp(x)>-bounds.min()*1.1:
                x *= delta_x

            box[1] = x

            dx = binary_search(lambda x: -dlogp(box[0]+x),
                                np.array([0,box[1]-box[0]]),bounds)
            log_right_point = box[0]+dx

            return log_right_point
        else:
            return x
    
    def find_points(dlogp,x0,bounds=np.array([.1,1.]),delta_x=1.1):

        if dlogp(x0)>0:

            #find left side
            log_left_point = find_left_point(dlogp,x0,bounds,delta_x)
            log_right_point = find_right_point(dlogp,log_left_point,bounds,delta_x)

        else:

            log_right_point = find_right_point(dlogp,x0,bounds,delta_x)
            log_left_point = find_left_point(dlogp,log_right_point,bounds,delta_x)

        return np.array([log_left_point,log_right_point])
    
    def get_edges(xs,f,dxs):
        edges = np.zeros(xs.size-1)
        for i in np.arange(xs.size-1):
            edges[i] = (f(xs[i+1])-f(xs[i])+dxs[i]*xs[i]-dxs[i+1]*xs[i+1])/(dxs[i]-dxs[i+1])
 
        
        return edges
    
    def get_zs(f,xs,dxs,offset):
        edges = get_edges(xs,f,dxs)

        log_zs = np.zeros(edges.size+1)
        if dxs[0]>0:
            log_zs[0] = -offset+f(xs[0])-dxs[0]*xs[0]-np.log(dxs[0])+dxs[0]*edges[0]+\
                    np.log(1-np.exp(-dxs[0]*edges[0]))
        else:
            log_zs[0] = -offset+f(xs[0])-dxs[0]*xs[0]-np.log(-dxs[0])+\
                    np.log(1-np.exp(dxs[0]*edges[0]))
        
        for i in np.arange(1,edges.size+1):
            if i==edges.size:
                log_zs[i] = -offset+f(xs[-1])-dxs[-1]*(xs[-1]-edges[-1])-np.log(-dxs[-1])
            else:
                foo = np.array([dxs[i]*edges[i],dxs[i]*edges[i-1]])

                log_zs[i] = -np.log(np.abs(dxs[i]))-offset+f(xs[i])-dxs[i]*xs[i]+\
                    foo.max()+np.log(1-np.exp(foo.min()-foo.max()))

        z = np.where(np.random.multinomial(1,softmax(log_zs))==1)[0][0]
        
        return log_zs
    
    def sample(log_zs,dxs,edges):

        z = np.where(np.random.multinomial(1,softmax(log_zs))==1)[0][0]

        if z==0:
            if dxs[0]>0:
                
                u = np.random.rand()
                samp = edges[0]+np.log(u+(1-u)*np.exp(-dxs[0]*edges[0]))/dxs[0]
            else:
                u = np.random.rand()
                samp = np.log(1+u*(np.exp(dxs[0]*edges[0])-1))/dxs[0]
                
        elif z==(zs.size-1):
            samp = edges[-1] + 1/dxs[-1]*np.log(1-np.random.rand())
        else:
            if dxs[z]<0:
                pmax = 1-np.exp(dxs[z]*np.array(edges[z]-edges[z-1]))
                samp = edges[z-1] + 1/dxs[z]*np.log(1-np.random.rand()*pmax)
            else:
                pmax = 1-np.exp(-dxs[z]*np.array(edges[z]-edges[z-1]))
                samp = edges[z] + 1/dxs[z]*np.log(1-np.random.rand()*pmax)
        return samp

    def update_approx(f,df,xs):
        dxs = df(xs)

        edges = get_edges(xs,f,dxs)
        max_idx = np.argmax(f(.5*(xs[1:]+xs[:-1])))
        
        offset=0
        zs = get_zs(f,xs,dxs,offset)
        return zs,dxs,edges
    
    
    if df(1)>=lbounds.min():
        #left point existing means right point exists
        right_point = find_right_point(df,1.,rbounds,delta_x=1.1)

        left_point = find_left_point(df,.5*right_point,lbounds,delta_x=1.1)
    else:
        if df(1e-2)>-rbounds.max():
            #find right point as usual
            right_point = find_right_point(df,1.,rbounds,delta_x=1.1)
            left_point = .5*right_point
        else:
            right_point = 1.
            left_point = .5
            
    offset=0
    xs = np.array([left_point,right_point])
        
    f = lambda x: ff(x)-ff(xs.mean())

    zs,dxs,edges = update_approx(f,df,xs)
    
    samps = np.zeros(nsamps)
    i=0
    while i<nsamps:
    
        prop = sample(zs,dxs,edges)
        piece_idx = np.digitize(prop,edges)

        log_acc_prob = f(prop)-(f(xs[piece_idx])+dxs[piece_idx]*(prop-xs[piece_idx]))
        
        if np.log(np.random.rand())<log_acc_prob:
            samps[i]=prop
            i = i+1
            
        else:
            xs = np.sort(np.concatenate([xs,np.array([prop])]))
            zs,dxs,edges = update_approx(f,df,xs)
            
    if nsamps==1:
        return samps[0]
    else:
        return samps
    

def sample_Uk(rated_idx,labels,U,V,t,k,sign,lamba=1):
    K = V.shape[1]
    M,L = U[:,:,:,0].shape[:2]

    for m in np.arange(M):
        for l in np.arange(L):
            idx = np.intersect1d(np.where(t==l),rated_idx[m])
            for ll in np.arange(L):
                if (sign=='pos' and ll==l) or (sign=='neg' and l!=ll) or sign==None:
                    idxx = rated_idx[m][np.logical_and(t[rated_idx[m]]==l,labels[m]==ll)]
                    UV = np.dot(U[m,l],V[idx].T).T
                    Vsum = V[idxx,k].sum()

                    def f(xx):
                        if np.isscalar(xx):
                            foo = UV.copy()
                            foo[:,ll] += (xx-U[m,l,ll,k])*V[idx,k]
                            return xx*Vsum-logsumexp(foo,-1).sum()-.5*lamba*xx**2
                        else:
                            foo = np.repeat(UV[None,:,:],xx.size,axis=0)
                            foo[:,:,ll] += (xx[:,None]-U[m,l,ll,k])*V[idx,k][None,:]        
                            return xx*Vsum-logsumexp(foo,-1).sum(-1)-.5*lamba*xx**2


                    def df(xx):
                        if np.isscalar(xx):
                            foo = UV.copy()
                            foo[:,ll] += (xx-U[m,l,ll,k])*V[idx,k]
                            return Vsum-(V[idx,k]*softmax(foo,-1)[:,ll]).sum()-lamba*xx
                        else:
                            foo = np.repeat(UV[None,:,:],xx.size,axis=0)
                            foo[:,:,ll] += (xx[:,None]-U[m,l,ll,k])*V[idx,k][None,:]
                            return Vsum-(V[idx,k][None,:]*softmax(foo,-1)[:,:,ll]).sum(-1)-lamba*xx

                    U[m,l,ll,k] = ars_nonneg(f,df,U[m,l,ll,k])
                    
    return U[:,:,:,k]