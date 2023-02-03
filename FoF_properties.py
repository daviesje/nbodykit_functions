import argparse
import bigfile as bf
import numpy as np
from os.path import exists
from astropy import cosmology, constants as C, units as U
from scipy import integrate
from scipy.stats import binned_statistic as binstat
from scipy.signal import fftconvolve
from mpi4py import MPI

import nbodykit
nbodykit.use_mpi()
nbodykit.set_options(dask_chunk_size=1024*1024*8)
nbodykit.setup_logging(log_level='info')
from nbodykit.lab import BigFileCatalog
import dask.array as da

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["text.usetex"] = False

from astropy import cosmology, constants as C, units as U
#Astrid cosmology
h = 0.7186
Om0 = 0.2814
Ob0 = 0.0464
m_nu = [0., 0., 0.]*U.Unit('eV')
Tcmb0 = 2.7255
Ode0 = 0.7186
cosmo = cosmology.FlatLambdaCDM(H0=h*100,Om0=Om0,Ob0=Ob0,Tcmb0=Tcmb0,m_nu=m_nu)

def percentilefunc(x,p=np.array([2.5,16,50,84,97.5]),axis=None):
    return np.nanpercentile(x,p,axis=axis)

def momentfunc(x,axis=None):
    return np.array([np.nanmean(x,axis=axis),np.nanstd(x,axis=axis),np.nanvar(x,axis=axis)])

#power law means, scalings, lognormal scatters for comparison with scaling relations
f_shmr, a_shmr, s_shmr =  0.05, 0.5, 0.6
t_ssfr, s_ssfr = 0.5, 0.5

#Lognormal models
def shmr_model(hm,f=f_shmr,a=a_shmr,s=s_shmr):
    mean = f * (hm / 1e10)**a_shmr
    median = mean * np.exp(-s*s/2)

    sigp = median * np.exp(s)
    sigm = median / np.exp(s)

    return mean, sigm, median, sigp

def ssfr_model(sm,t=t_ssfr,s=s_ssfr,z=0):
    mean = sm * cosmo.H(z) / t
    median = mean * np.exp(-s*s/2)

    sigp = median * np.exp(s)
    sigm = median / np.exp(s)

    return mean, sigm, median, sigp

def parse_property(dset, prop, **kwargs):
    #special cases where we need a slice, unit changes or transformation
    redshift = 1/dset.attrs['Time'] - 1
    h = dset.attrs["HubbleParam"]
    if prop == 'StellarMass':
        arr = dset['MassByType'][:,4] * 1e10 / h
    elif prop == 'lnStellarMass':
        arr = da.log(dset['MassByType'][:,4] * 1e10 / h)
    elif prop == 'GasMass':
        arr = dset['MassByType'][:,0] * 1e10 / h
    elif prop == 'Mass':
        arr = dset['Mass'] * 1e10 / h
    elif prop == 'lnMass':
        arr = da.log(dset['Mass'] * 1e10 / h)
    elif prop == 'DzReion':
        gpos = ((dset['MassCenterPosition']/kwargs['reionres']).astype(int))
        arr = da.from_array(kwargs['reiongrid'][gpos[:,0],gpos[:,1],gpos[:,2]]) - redshift
    elif prop == 'zReion':
        gpos = ((dset['MassCenterPosition']/kwargs['reionres']).astype(int))
        arr = da.from_array(kwargs['reiongrid'][gpos[:,0],gpos[:,1],gpos[:,2]])
    elif prop == 'PosX':
        arr = dset['MassCenterPosition'][:,0]
    elif prop == 'PosY':
        arr = dset['MassCenterPosition'][:,1]
    elif prop == 'PosZ':
        arr = dset['MassCenterPosition'][:,2]
    elif prop == 'SFRonH':
        arr = dset['StarFormationRate'] / cosmo.H(redshift).value #TODO: use the read little h
    elif prop == 'lnSFRonH':
        arr = da.log(dset['StarFormationRate'] / cosmo.H(redshift).value) #TODO: use the read little h
    elif prop == 'FStar':
        arr = dset['MassByType'][:,4] / dset['MassByType'][:,0]
    elif prop == 'FStarM':
        arr = dset['MassByType'][:,4] / dset['Mass']
    elif prop == 'FGas':
        arr = dset['MassByType'][:,0] / dset['Mass']
    elif prop == "tSFR":
        arr = dset['StarFormationRate'] / (dset['MassByType'][:,4] * 1e10 / h) / cosmo.H(redshift).to('yr-1').value #TODO: use the read little h
    elif prop == 'lnFStar':
        arr = da.log(dset['MassByType'][:,4] / dset['Mass'])
    elif prop == "lntSFR":
        arr = da.log(dset['StarFormationRate'] / (dset['MassByType'][:,4] * 1e10 / h) / cosmo.H(redshift).to('yr-1').value) #TODO: use the read little h
    elif prop == 'S2lnFStar':
        lnfs = da.log(dset['MassByType'][:,4] / dset['Mass'])
        sel = da.isfinite(lnfs)
        avg = da.mean(lnfs[sel])
        arr = (lnfs - avg)*(lnfs - avg) # variance of each sample
    elif prop == "S2lntSFR":
        lnt = da.log(dset['StarFormationRate'] / (dset['MassByType'][:,4] * 1e10 / h) / cosmo.H(redshift).to('yr-1').value)
        sel = da.isfinite(lnt)
        avg = da.mean(lnt[sel])
        arr = (lnt - avg)*(lnt - avg)
    else:
        try:
            arr = dset[prop]
        except:
            logger.warning(f"cannot find {prop}")
            return None

    return arr

#collect properties of groups at one snapshot and return all in memory
def read_fof_snap(fname,props,gather_out=True,**kwargs):
    dset = BigFileCatalog(f'{fname}',dataset='FOFGroups/')

    redshift = 1/dset.attrs['Time'] - 1

    comm = dset.comm
    nrank = comm.Get_size()
    if gather_out:
        sendcounts = comm.allgather(dset.size)
        if comm.rank == 0:
            result_all = np.zeros((len(props),dset.csize),dtype='f8')
            recvbuf = np.zeros(dset.csize,dtype='f8')
            logger.info('reading snap %s z=%.2f',fname,redshift)
            logger.info(f"cat sizes {sendcounts} csize {dset.csize}")
        else:
            result_all = None
            recvbuf = None
    else:
        result_all = np.zeros((len(props),dset.size),dtype='f8')

    for j,prop in enumerate(props):
        buf = parse_property(dset,prop,**kwargs)
        if buf is None:
            if comm.rank == 0:
                logger.warning('cannot find %s in %s',prop,fname)
            continue
        result_local = buf.compute().astype('f8')

        if comm.rank == 0:
            logger.info(f'{prop} has range ({np.nanmin(result_local)}'
                           f',{np.nanmean(result_local)},{np.nanmax(result_local)})')

        if gather_out:
            comm.Gatherv(result_local,(recvbuf,sendcounts),root=0)
            if comm.rank == 0:
                result_all[j,...] = recvbuf
        else:
            result_all[j,...] = result_local

    return result_all

#collect properties of groups with particular Black hole particles in them
#TODO: add dask back in (compute after rather than before) for large datasets
def read_fof_ids(fname,props,ids,reiongrid=None,reionres=None):
    dset_bh = BigFileCatalog(f'{fname}',dataset='5/')

    redshift = 1/dset_bh.attrs['Time'] - 1

    comm = dset_bh.comm
    nrank = comm.Get_size()
    result = np.zeros((len(props),ids.size))

    id_buf = dset_bh['ID'].compute()
    gid_buf = dset_bh['GroupID'].compute()
    #check for no galaxies since zero length arrays mess things up
    check_local = np.any(np.isin(ids,id_buf))
    if check_local:
        #searchsorted places the local BH ids in the global BH id array
        global_idx = np.searchsorted(ids,id_buf)
        #mask for ids not in global array (If refsnap < snap)
        #also for BH not in groups (check if this removes swallowed)
        mask_g = (ids[global_idx] == id_buf) & (gid_buf > 0)
        global_idx = global_idx[mask_g]
        id_buf = id_buf[mask_g]
        gid_buf = gid_buf[mask_g]
        
        if comm.rank == 0:
            logger.info('wanted ids: %s from %d',ids.size,dset_bh.csize)
        #logger.info('rank %02d has %s ids out of %s',comm.rank,global_idx.shape,id_buf.shape)

        dset_fof = BigFileCatalog(f'{fname}',dataset='FOFGroups/')
        #this is already sorted
        
        fof_id = dset_fof['GroupID'].compute()
        #sendcounts = comm.allgather(fof_id.size)
        #comm.Allgatherv(fof_id,(fof_id,sendcounts))
        fof_id = np.concatenate(comm.allgather(fof_id),axis=0)
        if np.any(fof_id[1:] < fof_id[:-1]):
            logger.error('fofid not sorted')
            quit()

        #This searchsorted places the fof GIDs in the BH array
        fof_idx = np.searchsorted(fof_id,gid_buf)
        #mask for BH groups not in the FOF list (only happens if BH has no group)
        mask_f = (fof_id[fof_idx] == gid_buf)
        fof_idx = fof_idx[mask_f]
        id_buf = id_buf[mask_f]
        gid_buf = gid_buf[mask_f]
        global_idx = global_idx[mask_f]
        '''
        if comm.rank == comm.size - 1:
            logger.info(f'target BH shape {ids.shape}')
            logger.info(f'snap BH shape {id_buf.shape}')
            logger.info(f'global idx shape {global_idx.shape}')
            logger.info(f'fof idx shape {fof_idx.shape}')
            logger.info(f'fof shape {fof_id.shape}')

            logger.info(f'first few BH groups {gid_buf[:5]}')
            logger.info(f'placed in fof arr at {fof_idx[:5]}')
            logger.info(f'which has GID {fof_id[fof_idx[:5]]}')
            logger.info(f'first few BH PID {id_buf[:5]}')
            logger.info(f'placed in ID arr at {global_idx[:5]}')
            logger.info(f'which has PID {ids[global_idx[:5]]}')
        '''
        for j,prop in enumerate(props):
            data = parse_property(dset_fof,prop,reiongrid=reiongrid,reionres=reionres).compute()
            data = np.concatenate(comm.allgather(data),axis=0)
            if data is not None:
                #data[fof_idx] and result[global_idx] should both be
                #in the space of the current snapshot BH particles
                result[j,global_idx] = data[fof_idx]

    #Since each BH should only exist on one rank, summing will place everything correctly
    result = comm.reduce(result,op=MPI.SUM,root=0)

    if comm.rank == 0:
        return result
    else:
        return None


#build a list of BH particles at one snapshot and follow them across time
def collect_fof_history(dname,snapshot_list,refsnap,props):
    #load reference snapshot to get all BH GroupIDs we care about
    dset = BigFileCatalog(f'{dname}/PIG_{refsnap:03d}',dataset='5/')
    #we need the list of BH particle IDs sorted (?) and on all ranks
    gid = dset['ID'].compute()
    gid = np.concatenate(dset.comm.allgather(gid),axis=0)
    idxsort = np.argsort(gid)
    gid = gid[idxsort]

    comm = MPI.COMM_WORLD
    nrank = comm.Get_size()

    if comm.rank == 0:
        out_array = np.zeros((snapshot_list.size,len(props),gid.size))
    else:
        out_array = None

    #snapshot mask to account for missing snapshots in the data itself
    snap_mask = np.ones(len(snapshot_list),dtype=bool)

    for i, snap in enumerate(snapshot_list):
        #read in the particle file
        fofname = f'{dname}/PIG_{snap:03d}/'

        if not exists(fofname):
            snap_mask[i] = False
            continue

        buf = read_fof_ids(fname=fofname,props=props,ids=gid)
        if comm.rank == 0:
            out_array[i,...] = buf
            logger.info("snapshot %s done", snap)
    
    return out_array

def fof_autocorr(dname, z_list, refsnap, props=['GroupID','Mass','StellarMass','StarFormationRate']):
    nprops = len(props)
    p_offset = 0
    if 'GroupID' not in props:
        props = ['GroupID'] + props
        p_offset = 1

    comm = MPI.COMM_WORLD
    snapshot_list, z_snap = find_redshifts(dname,z_list)
    #(snaps,props,ids)
    data = collect_fof_history(dname,snapshot_list,refsnap,props)
    if comm.rank == 0:
        dz = np.fabs(np.mean(np.diff(z_snap)))
        dz_base = np.linspace(-(dz*(z_snap.size-1)),dz*(z_snap.size-1),num=2*z_snap.size-1)

        if np.any(np.diff(z_snap) != z_snap[1]-z_snap[0]):
            logger.info("redshift array not unifrom")
            logger.info(f"{z_snap},{dz_base}")

        #TODO: cull by uniqueness to remove identical groups
        #if two rows have the same GroupID at one snapshot,
        #the other props should also be identical

        #we only want galaxies that exist, have stars and SFR for the whole timespan
        data = data[...,np.all(data>0,axis=(0,1))]
        #normalise data for each gal/prop before autocorrelation
        data_normed = (data-np.nanmean(data,axis=0))/np.nanstd(data,axis=0)
        
        #we use fftconvolve instead of correlate since it has the axes option
        corr = fftconvolve(data_normed,data_normed[::-1,...],axes=0)

        #take moments/percentiles over the bh IDs
        corr_percentiles = percentilefunc(corr,axis=-1)
        corr_moments = momentfunc(corr,axis=-1)

        print(corr_moments.shape,corr_percentiles.shape)

        fig,axs = plt.subplots(nrows=nprops,ncols=2,figsize=(8,4))
        rand_idx = np.random.randint(0,corr.shape[-1],size=24)

        for i in range(nprops):
            idx = i+p_offset
            axs[i,0].fill_between(dz_base,corr_percentiles[0,:,idx],corr_percentiles[-1,:,idx],color='k',alpha=0.15)
            axs[i,0].plot(dz_base,corr_moments[0,:,idx],'k-')
            axs[i,0].plot(dz_base,corr[:,idx,rand_idx],'k:',linewidth=0.25)
            axs[i,0].grid()
            axs[i,0].set_ylim([-5,15])
            axs[i,0].set_xlim([dz_base.min(),dz_base.max()])
            if i != nprops - 1:
                axs[i,0].tick_params(bottom=False,labelbottom=False)
                axs[i,1].tick_params(bottom=False,labelbottom=False)
            else:
                axs[i,0].set_xlabel('dz')
                axs[i,1].set_xlabel('z')
            axs[i,0].set_ylabel(f'<XX*> {props[idx]}')

            axs[i,1].plot(z_snap,data_normed[:,idx,rand_idx],'r',linewidth=0.5)
            axs[i,1].grid()
            axs[i,1].set_ylabel(f'{props[idx]}(N)')
            axs[i,1].set_yscale('log')
            axs[i,1].set_ylim(2e0,1e-1)
            
        fig.subplots_adjust(top=0.99,bottom=0.13,left=0.07,right=0.99,hspace=0,wspace=0.13)
        fig.savefig('./autocorr.png')

    return


#the goal here is to provide a model for Mass, Stellarmass, SFR and compare to the simulation
def fof_compare_model():

    return

def fof_cov(fname,props):
    dset = BigFileCatalog(f'{fname}',dataset='FOFGroups/')

    redshift = 1/dset.attrs['Time'] - 1

    data = np.stack([parse_property(dset,prop,redshift=redshift).compute() for prop in props],axis=0)

    sel = ((dset['LengthByType'][:,1].compute() > 100) & np.all(np.isfinite(data),axis=0)) #select out tiny groups and zero star groups
    print(f"selecting {sel.sum()} elements from {sel.shape}, first 3 {data[:,:3]} last 3 {data[:,-3:]}")
    #print(data)
    data = data[:,sel]
    #print(data[:,:10],data[:,-10:])

    cov = np.cov(data)

    return cov

def plot_cov(dname,snaps,redshifts,outname,mode="cov"):
    props = ['lnMass','lnStellarMass','lnSFRonH','lnFStar','lntSFR','S2lnFStar','S2lntSFR']
    
    #setup the figure
    aspect_target = 16./9.
    aspect = 1.
    nrow = 1
    ncol = 1
    for i in range(len(snaps)):
        if i+1 > nrow*ncol:
            if aspect > aspect_target:
                nrow += 1
            else:
                ncol += 1
            aspect = ncol / nrow

    fw = 16
    fh = (fw/aspect)/1.2 #a little more width for cbar
    fig, axes = plt.subplots(nrows=nrow,ncols=ncol,sharex='none',sharey='none',figsize=(fw,fh))
    if len(snaps) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i,snap in enumerate(snaps):
        fofname = f'{dname}/PIG_{snap:03d}/'
        logger.info(f'starting {fofname}')
        cov = fof_cov(fofname,props)
        if mode == "coef":
            cov = cov / np.sqrt(np.diag(cov)[None,:]*np.diag(cov)[:,None])
        elif mode == "slope":
            cov = cov / np.diag(cov)[None,:]

        if i == 0:
            if mode == 'coef':
                norm = matplotlib.colors.Normalize(vmin=-1,vmax=1)
                cmap = "RdBu"
            else:
                norm = matplotlib.colors.Normalize(vmin=-cov.max(),vmax=cov.max())
                cmap = "RdBu"

        im = axes[i].imshow(cov,cmap=cmap,norm=norm)
        axes[i].set_title(f'z={redshifts[i]:.2f}')

        axes[i].set_xticks(np.arange(len(props)))
        axes[i].set_yticks(np.arange(len(props)))
        axes[i].set_xticklabels(props)
        axes[i].set_yticklabels(props)

        for j in range(len(props)):
            for k in range(len(props)):
                text = axes[i].text(k, j, f'{cov[j, k]:.2e}', ha="center", va="center", color="k")

    fig.subplots_adjust(top=0.9,bottom=0.1,left=0.15,right=0.99)
    plt.colorbar(im,ax=axes)

    fig.savefig(outname)

def binned_fofstats(fname,prop,binprop,bins=None,binmode='hist',):
    dset = BigFileCatalog(f'{fname}',dataset='FOFGroups/')

    redshift = 1/dset.attrs['Time'] - 1

    y = parse_property(dset,prop,redshift=redshift)
    x = parse_property(dset,binprop,redshift=redshift)

    #bins should be a numpy array, percentiles can be a list
    out_percentiles = np.zeros((bins.size,len(global_percentiles)))
    out_moments = np.zeros((bins.size,3))
    out_hist = np.zeros((logdevbins.size - 1))

    #I can't find scipy binned_statistic in dask. Since the whole FOF catalogue will be ~1GB per column
    #I can get away with loading 2 columns using dask and calculating percentiles in memory
    #Although It would be nice to chunk/parallelise properly to extend this to work on particle stats

    #A more memory intensive but probably faster version would be to load everything first
    #and use scipy binned_statistic 7 times for the quantiles / mean / variance
    for i in range(bins.size):
        if  mode=='hist':
            if i == bins.size - 1:
                break
            xlow = bins[i]
            xhigh = bins[i+1]
        elif mode=='near':
            xlow = bins[i] / 1.3
            xhigh = bins[i] * 1.3

        xsel = (x >= xlow) & (x < xhigh) & (y > 0) # I think duty cycles should not be counted in the lognormal
        ybin = y[xsel].compute()

        #logger.info(f'bin {i}, {xlow:.2e}, {xhigh:.2e} has {ybin.size} elements')

        #if there are no fofgroups in the bin or all groups in the bin have zero property
        if ybin.size == 0 or np.amax(ybin) == 0:
            continue

        out_percentiles[i,:] = percentilefunc(ybin)
        out_moments[i,:] = momentfunc(np.log(ybin)) #I want everything to be lognormal so I check that here

    return out_percentiles, out_moments

def plot_fofstats(dname,snaps,redshifts,outname):
    props = ['StellarMass','SFRonH']
    binprops = ['Mass','StellarMass']

    colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8',]

    bins = []
    for j in range(len(binprops)):
        if binprops[j] == 'Mass':
            bins.append(np.logspace(9.5,12.5,num=24))
        elif binprops[j] == 'StellarMass':
            bins.append(np.logspace(6,10.5,num=24))
        else:
            logger.error(f'binning {binprops[j]} not implemented')
            quit()

    logdevbins = np.linspace(-2,2,num=16)
    dlogd = (logdevbins[1:]) - (logdevbins[:-1])
    centresd = (logdevbins[:-1] + dlogd/2)

    #setup the figure
    #we have n sets of properties, and 2 kinds of plots (means, scatters)
    nrow = 3
    ncol = len(props)
    fig, axes = plt.subplots(nrows=nrow,ncols=ncol,sharex='none',sharey='none',figsize=(12,8*(nrow/ncol)))
    
    for i, snap in enumerate(snaps):
        #read in the particle file
        fofname = f'{dname}/PIG_{snap:03d}/'
        logger.info(f'starting {fofname}')
        for j in range(len(props)):
            logger.info(f'binning {props[j]} by {binprops[j]}')
            p,m,h = binned_fofstats(fofname,props[j],binprops[j],bins=bins[j],logdevbins=logdevbins)
            h = h / dlogd
            h = h / np.amax(h)

            #quantile plot (row 0)
            axes[0,j].loglog(bins[j],p[:,2],color=colors[i],linestyle='-',linewidth=2,label=f'z={redshifts[i]:.1f}')
            axes[0,j].loglog(bins[j],p[:,1],color=colors[i],linestyle='--',linewidth=1,alpha=0.5)
            axes[0,j].loglog(bins[j],p[:,3],color=colors[i],linestyle='--',linewidth=1,alpha=0.5)
            axes[0,j].set_ylabel(props[j])
            axes[0,j].set_xlabel(binprops[j])
            axes[0,j].grid()

            #stdev plot (row 1)
            axes[1,j].semilogx(bins[j],m[:,1],color=colors[i],linestyle='-',linewidth=2)
            axes[1,j].semilogx(bins[j],np.log(p[:,3]/p[:,2]),color=colors[i],linestyle='--',linewidth=1,alpha=0.5)
            axes[1,j].semilogx(bins[j],np.log(p[:,2]/p[:,1]),color=colors[i],linestyle=':',linewidth=1,alpha=0.5)
            axes[1,j].set_xlabel(binprops[j])
            axes[1,j].grid()

            axes[2,j].plot(centresd,h,color=colors[i],linestyle='-',linewidth=2)
            axes[2,j].set_xlabel(f'{props[j]} / median')

            #the sigma from m is the stdev of log(X)
            sigma = np.nanmean(m[:,1][m[:,1] > 0]) #guess at a single sigma by averaging across bins where there are halos
            model = np.exp(-((centresd)**2/(2*sigma*sigma))) #normal distribution of log(X/median)
            axes[2,j].plot(centresd,model,color=colors[i],linestyle='--',linewidth=1,alpha=0.5)
            axes[2,j].grid()

    axes[0,0].legend()
    axes[1,0].set_ylabel('Log Sigma')
    axes[2,0].set_ylabel(f'dN/dlog (ratio)')
    fig.savefig(outname)

    return fig

#read a 
def fof_ndhist(dname,props):
    logger.info('Making %s histogram from %s',props,dname)
    data = read_fof_snap(dname,props)

    bins = []
    for i,p in enumerate(props):
        if p == 'Mass':
            bins.append(np.logspace(9.5,12.5,num=21))
        elif p == 'StellarMass':
            bins.append(np.logspace(6,10.5,num=20))
        elif p == 'StarFormationRate':
            bins.append(np.logspace(-3,3,num=19))
        else:
            bins.append(np.logspace(np.log10(data[i,:].min()),np.log10(data[i,:].max())),num=24)

    print(len(bins),data.shape)

    hist,edges = np.histogramdd(data.T,bins)

    return hist,edges

def fof_3dhistplot(dname,outname,props=['Mass','StellarMass','StarFormationRate']):
    if(len(props)!=3):
        logger.error('only use for 3 properties')
        return
    hist,edges = fof_ndhist(dname,props)

    centres = []
    #logspace edges to centres
    for e in edges:
        w = np.log(e[1:]) - np.log(e[:-1])
        centres.append(e[:-1] * np.exp(w/2))

    total = hist.sum()

    hist2d_0 = hist.sum(axis=0)
    #srt = np.sort(hist2d_0,axis=None)[::-1]
    #cum = np.cumsum(srt)/total
    #z_0 = [srt[np.argmax(cum > 0.95)],srt[np.argmax(cum > 0.68)]]

    corrcoef_0 = np.zeros_like(centres[0])
    corrcoef_1 = np.zeros_like(centres[1])
    
    crd_0 = np.stack(np.meshgrid(centres[1],centres[2],indexing='ij'),axis=-1).reshape(-1,2)
    crd_1 = np.stack(np.meshgrid(centres[0],centres[2],indexing='ij'),axis=-1).reshape(-1,2)
    logger.info('CRDS %s %s',crd_0.shape,crd_0[:5,:])
    logger.info('CEN %s %s',centres[1][:5],centres[2][:5])
    for i,(mh,ms) in enumerate(zip(centres[0],centres[1])):        
        logger.info('%s',i)
        buf = hist[i,...].reshape(-1)
        
        if i==0:
            logger.info('BUFFER %s %s',buf.shape,buf[:5])
            logger.info('HIST   %s',hist[i,...][:5,:5])
        #if there are no galaxies, don't bother
        if buf.sum() == 0:
            corrcoef_0[i] = 0.
        else:
            cov = np.cov(np.log(crd_0).T,fweights=buf) #weighted by the histogram
            corrcoef_0[i] = cov[0,1]/np.sqrt(cov[1,1]*cov[0,0])

        buf = hist[:,i,:].reshape(-1)
        if buf.sum() == 0:
            corrcoef_1[i] = 0.
        else:
            cov = np.cov(np.log10(crd_1).T,fweights=buf) #weighted by the histogram
            corrcoef_1[i] = cov[0,1]/np.sqrt(cov[1,1]*cov[0,0])

    avgbin_0 = np.sum(hist * np.log10(centres[0][:,None,None]),axis=0)/hist2d_0 #average SFR in (Mh,M*) bins
    avgbin_0[avgbin_0!=avgbin_0] = 0
    stdbin_0 = np.sqrt(np.sum(hist * (np.log10(centres[0][:,None,None]) - avgbin_0[None,:,:])**2)/hist2d_0)
    stdbin_0[stdbin_0!=stdbin_0] = 0

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

    axs[0].semilogx(centres[0],corrcoef_0)
    axs[0].set_xlabel(props[0],fontsize=10)
    axs[0].set_ylabel(f'Corr({props[1][:5]},{props[2][:5]})(log)')
    axs[0].set_ylim(0,1)
    axs[1].semilogx(centres[1],corrcoef_1)
    axs[1].set_xlabel(props[1],fontsize=10)
    axs[1].set_ylabel(f'Corr({props[0][:5]},{props[2][:5]})(log)')
    axs[1].set_ylim(0,1)

    fig.subplots_adjust(wspace=0.4,top=0.97,left=0.08,right=0.95,bottom=0.2)
    fig.savefig(outname+'corr.png')

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

    im = axs[0].imshow(avgbin_0.T,origin='lower',cmap='viridis'
                ,extent=[np.log10(edges[0]).min(),np.log10(edges[0]).max()
                ,np.log10(edges[1]).min(),np.log10(edges[1]).max()]
                ,norm=matplotlib.colors.Normalize(vmin=np.log10(edges[0].min()),vmax=np.log10(edges[0].max()))
                ,aspect='auto')
    plt.colorbar(im,ax=axs[0])

    #axs[0].contour(np.log10(centres[0]),np.log10(centres[1]),hist2d_2.T,levels=z_2,colors='white')
    
    axs[0].set_xlabel(props[1],fontsize=10)
    axs[0].set_ylabel(props[2],fontsize=10)
    axs[0].set_title(f'<{props[0]}>')
    print(avgbin_0)
    print(stdbin_0)
    im = axs[1].imshow(stdbin_0.T,origin='lower',cmap='cividis'
                ,extent=[np.log10(edges[1]).min(),np.log10(edges[1]).max()
                ,np.log10(edges[2]).min(),np.log10(edges[2]).max()]
                ,norm=matplotlib.colors.Normalize(vmin=0,vmax=2)
                ,aspect='auto')
    plt.colorbar(im,ax=axs[1])

    #axs[0].contour(np.log10(centres[0]),np.log10(centres[1]),hist2d_2.T,levels=z_2,colors='white')
    
    axs[1].set_xlabel(props[1],fontsize=10)
    axs[1].set_ylabel(props[2],fontsize=10)
    axs[1].set_title(f'std({props[0]})')
    
    fig.subplots_adjust(wspace=0.4,top=0.9,left=0.08,right=0.95,bottom=0.1)
    fig.savefig(outname+'hist.png')

#make a plot of galaxy properties over time grouped by halo mass & z_r
def plot_zr(datadir,redshifts,properties,zr_targets,mass_targets,reion_path=None):
    snaps,z_arr = find_redshifts(datadir,redshifts)
    comm = MPI.COMM_WORLD
    
    properties = ['Mass'] + properties + ['zReion']
    
    #TODO: generalise labels & limits to all properties
    plabels = [r'$ M_{\mathrm{gas}}/M_{\mathrm{halo}} $'
            ,r'$ M_{*}/M_{\mathrm{gas}} $'
            ,r'$\mathrm{SFR} / M_{*} (t_H^{-1})$']
    #ylims = np.array[(0.13,0.17),(1e-3,2e-2),(1e0,2e1)]
    mlabels = [rf'$M = {{{m:.1e}}}$' for m in mass_targets]
    zlabels = [rf'$z_r = {{{z:.2f}}}$' for z in zr_targets]

    #all ranks need the box
    rpath,rset = reion_path.rsplit('/',1)
    reion_file = bf.File(rpath)
    dset = reion_file[rset]
    nmesh = int(dset.attrs['Nmesh'])
    reion_res = dset.attrs['BoxSize'] / nmesh * 1000. #mpc to kpc
    reion_grid = dset.read(0,dset.size)
    reion_grid = reion_grid.reshape(nmesh,nmesh,nmesh)
    reion_file.close()
    
    dz_plot_data = np.zeros((len(mass_targets),len(zr_targets),len(properties),z_arr.size),dtype='f8')
    dz_plot_counts = np.zeros((len(mass_targets),len(zr_targets),1,z_arr.size),dtype='f8')
    
    for i,z in enumerate(z_arr):
        if comm.rank == 0:
            logger.info(f"starting {snaps[i]},z={z:.2f}")

        fname = f'{datadir}/PIG_{int(snaps[i]):03d}'

        fofdata = read_fof_snap(fname,properties,gather_out=False,reiongrid=reion_grid,reionres=reion_res,redshift=z)
        
        condition_mass = [((fofdata[0,:] < mm*1.3)*(fofdata[0,:] > mm / 1.3)) for mm in mass_targets]
        condition_zr = [((fofdata[-1,:] < zr + 0.5)*(fofdata[-1,:] > zr - 0.5)) for zr in zr_targets]
        nonzero = np.all(fofdata > 0,axis=0)
        for im,cm in enumerate(condition_mass):
            for iz,cz in enumerate(condition_zr):
                sel = cm & cz & nonzero
                dz_plot_data[im,iz,:,i] += fofdata[:,sel].sum(axis=-1)
                dz_plot_counts[im,iz,:,i] += fofdata[0,sel].size

    dz_plot_data = comm.allreduce(dz_plot_data,op=MPI.SUM)
    dz_plot_counts = comm.allreduce(dz_plot_counts,op=MPI.SUM)
    dz_plot_counts[dz_plot_counts<=0] = 1 #it will be zero anyway
    dz_plot_data /= dz_plot_counts

    if comm.rank == 0:
        nrows = len(properties)-2
        ncols = len(mass_targets)
        fsize = (8,nrows/ncols*8)
        gs = gridspec.GridSpec(nrows=nrows,ncols=ncols)

        logger.info(f'{dz_plot_data.min(axis=(0,1,3))},{dz_plot_data.mean(axis=(0,1,3))},{dz_plot_data.max(axis=(0,1,3))}')

        fig = plt.figure(figsize=fsize)
        fig2,axs2 = plt.subplots(ncols=1,nrows=nrows,figsize=(4,1.5*nrows))
        for k in range(ncols):
            for j in range(nrows):
                ax = fig.add_subplot(gs[j,k])
                if j == 0:
                    ax.set_title(mlabels[k])
                    
                if j == nrows-1:
                    ax.set_xlabel(r'z')
                else:
                    ax.tick_params(labelbottom=False)
                
                if k == 0:
                    ax.set_ylabel(plabels[j])
                #else:
                    #ax.tick_params(labelleft=False)

                [ax.axvline(z,linestyle=':',color=f'C{i}',linewidth=2,alpha=0.5,zorder=1) for i,z in enumerate(zr_targets)]
                [ax.plot(z_arr,dz_plot_data[k,i,j+1,:],label=zlabels[i],zorder=2) for i,z in enumerate(zr_targets)]
                
                ax.grid()
                ax.set_xlim([5,12])
                #ax.set_ylim(ylims[j,k])
                
                if j==0 and k==0:
                    ax.legend(loc='upper left')

        axs2[nrows-1].set_xlabel('z')
        axs2[0].set_title(mlabels[0])
        for j in range(nrows):
            ax2 = axs2[j]
            if j < nrows-1:
                ax.tick_params(labelbottom=False)
            ax2.set_ylabel(plabels[j])
            [ax2.axvline(z,linestyle=':',color=f'C{i}',linewidth=2,alpha=0.5,zorder=1) for i,z in enumerate(zr_targets)]
            [ax2.plot(z_arr,dz_plot_data[0,i,j+1,:],label=zlabels[i],zorder=2) for i,z in enumerate(zr_targets)]

        axs2[0].legend()
        fig.subplots_adjust(left=0.08,right=0.98,top=0.95,bottom=0.08,hspace=0,wspace=0.2)
        fig2.subplots_adjust(left=0.2,right=0.98,top=0.9,bottom=0.13,hspace=0)
        fig.savefig('./fof_grid_zr.png')
        fig2.savefig('./fof_zr_low.png')


#plot galaxy property correlations at one snapshot
def plot_gal_corr(data,pnames,xprop,yprop,outname,nbins=32):
    if len(xprop) != len(yprop):
        print('xprop and yprop must be the same length')
        return

    aspect_target = 4./3.
    aspect = 1.
    nrow = 1
    ncol = 1
    for i in range(len(xprop)):
        if i+1 > nrow*ncol:
            if aspect > aspect_target:
                nrow += 1
            else:
                ncol += 1
            aspect = ncol / nrow

    fw = 12
    fh = fw/aspect

    gs = gridspec.GridSpec(nrows=nrow,ncols=ncol)
    fig = plt.figure(figsize=(fw,fh))

    for i, (xp,yp) in enumerate(zip(xprop,yprop)):
        if yp not in pnames or xp not in pnames:
            print(f'cannot plot {yp}, {xp}')
            continue

        idx = pnames.index(xp)
        idy = pnames.index(yp)

        minx = np.amin(data[idx][data[idx,...] > 0])
        miny = np.amin(data[idy][data[idy,...] > 0])

        edgesx = np.logspace(np.log10(minx),np.log10(data[idx,...].max()),num=nbins)
        dlogx = np.log10(edgesx[1:]) - np.log10(edgesx[:-1])
        centresx = (edgesx[:-1] * np.exp(dlogx/2))
        edgesy = np.logspace(np.log10(miny),np.log10(data[idy,...].max()),num=nbins)
        dlogy = np.log10(edgesy[1:]) - np.log10(edgesy[:-1])
        centresy = (edgesy[:-1] * np.exp(dlogy/2))

        ax = fig.add_subplot(gs[i])
        if xp == yp:
            #X function (dN/dlogX) TODO: add volume
            dndx = np.histogram(data[idx,...],bins=edgesx)[0] / dlogx
            ax.plot(centresx,dndx)
            ax.set_xlabel(xp)
            ax.set_ylabel("dNdlogX (number)")

        elif xp in pnames:
            #correlation with another property        
            mean = binstat(data[idx,...],values=data[idy,...],bins=edgesx,statistic='mean')[0]
            ysig1p = binstat(data[idx,...],values=data[idy,...],bins=edgesx,statistic=sig1p)[0] - mean
            ysig1m = mean - binstat(data[idx,...],values=data[idy,...],bins=edgesx,statistic=sig1m)[0]
            ysig2p = binstat(data[idx,...],values=data[idy,...],bins=edgesx,statistic=sig2p)[0] - mean
            ysig2m = mean - binstat(data[idx,...],values=data[idy,...],bins=edgesx,statistic=sig2m)[0]

            ax.errorbar(centresx,mean,yerr=[ysig2m,ysig2p],marker='o',color='r',linestyle='none',elinewidth=1.5,capsize=3)
            ax.errorbar(centresx,mean,yerr=[ysig1m,ysig1p],marker='o',color='k',linestyle='none',elinewidth=1.5,capsize=3)
            ax.set_xlabel(xp)
            ax.set_ylabel(yp)
            ax.set_yscale('log')
            ax.set_xscale('log')

    fig.savefig(outname)
        
    return

def find_redshifts(datadir,redshift_targets):
    #find the closest redshifts
    time_list = np.loadtxt(f'{datadir}/Snapshots.txt', dtype=float, ndmin=2)
    snapshot_list = time_list[:,0].astype(int)
    redshift_list = 1/time_list[:,1] - 1
    snap_idx = np.unique(np.argmin(np.fabs(redshift_targets[:,None]-redshift_list[None,:]),axis=1))
    redshift_in = redshift_list[snap_idx]
    snap_in = snapshot_list[snap_idx]
    
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        logger.info(f'looking for {redshift_targets}, found {redshift_in} at {snap_in}')

    return snap_in, redshift_in    
