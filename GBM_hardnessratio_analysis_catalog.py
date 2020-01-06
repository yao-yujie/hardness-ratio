# General analyses of GBM catalog bursts 
# last modified: Apr. 29, 2019

from astropy.io import fits
from astropy.time import Time
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import h5py
from scipy import stats
import os
import sys
import re
from multiprocessing import Pool
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects as robjects
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
robjects.r("library(baseline)")
from xspec import *

#name=['bn180525151','bn170705115']
name = []
for line in open("sample.txt","r"):              
	name.append(line[:11])
nl=len(name)
databasedir='/home/yao/burstdownloadyears'
#databasedir='/home/yujie/downburstdata/data'

NaI=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
BGO=['b0','b1']
Det=['b0','b1','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']

ch1=3
ch2=124
ch3=25

ncore=10


time_slice=[]

epeak=[]
epeak_error_p=[]
epeak_error_n=[]


def get_usersjar():
    usersjar = "/home/yao/Software/users.jar"
    return usersjar

def query_fermigbrst(cdir='./'):
    fermigbrst = cdir+'/fermigbrst_test.txt'
    if not os.path.exists(fermigbrst):
        usersjar = get_usersjar()
        assert os.path.exists(usersjar), """'users.jar' is not available! 
            download users.jar at:
            https://heasarc.gsfc.nasa.gov/xamin/distrib/users.jar
            and update the path of usersjar in 'personal_settings.py'."""
        java_ready = os.system("java --version")
        assert not java_ready, """java not properly installed!
            Install Oracle Java 10 (JDK 10) in Ubuntu or Linux Mint from PPA
            $ sudo add-apt-repository ppa:linuxuprising/java
            $ sudo apt update
            $ sudo apt install oracle-java10-installer"""
        fields = ("trigger_name,t90,t90_error,t90_start,"
            "Flnc_Band_Epeak,scat_detector_mask")
        print('querying fermigbrst catalog using HEASARC-Xamin-users.jar ...')
        query_ready = os.system("java -jar "+usersjar+" table=fermigbrst fields="
                +fields+" sortvar=trigger_name output="+cdir+"/fermigbrst_test.txt")
        assert not query_ready, 'failed in querying fermigbrst catalog!'
        print('successful in querying fermigbrst catalog!')
    return fermigbrst

fermigbrst = query_fermigbrst()
df = pd.read_csv(fermigbrst,delimiter='|',header=0,skipfooter=3,engine='python')
trigger_name = df['trigger_name'].apply(lambda x:x.strip()).values
t90_str = df[df.columns[1]].apply(lambda x:x.strip()).values

t90_error_str = df[df.columns[2]].apply(lambda x:x.strip()).values
t90_start_str = df[df.columns[3]].apply(lambda x:x.strip()).values
Flnc_Band_Epeak_str = df[df.columns[4]].apply(lambda x:x.strip()).values
scat_detector_mask_str = df[df.columns[5]].apply(lambda x:x.strip()).values
burst_number = len(trigger_name)
print('burst_number = ',burst_number)

def norm_pvalue(sigma=2.0):
	p = stats.norm.cdf(sigma)-stats.norm.cdf(-sigma)
	return p


def write_phaI(spectrum_rate,bnname,det,t1,t2,outfile):
	header0=fits.Header()
	header0.append(('creator', 'Shao', 'The name who created this PHA file'))
	header0.append(('telescop', 'Fermi', 'Name of mission/satellite'))
	header0.append(('bnname', bnname, 'Burst Name'))
	header0.append(('t1', t1, 'Start time of the PHA slice'))
	header0.append(('t2', t2, 'End time of the PHA slice'))
	
	hdu0=fits.PrimaryHDU(header=header0)
	
	a1 = np.arange(128)
	col1 = fits.Column(name='CHANNEL', format='1I', array=a1)
	col2 = fits.Column(name='COUNTS', format='1D', unit='COUNTS', array=spectrum_rate)
	hdu1 = fits.BinTableHDU.from_columns([col1, col2])
	header=hdu1.header
	header.append(('extname', 'SPECTRUM', 'Name of this binary table extension'))
	header.append(('telescop', 'GLAST', 'Name of mission/satellite'))
	header.append(('instrume', 'GBM', 'Specific instrument used for observation'))
	header.append(('filter', 'None', 'The instrument filter in use (if any)'))
	header.append(('exposure', 1., 'Integration time in seconds'))
	header.append(('areascal', 1., 'Area scaling factor'))
	header.append(('backscal', 1., 'Background scaling factor'))
	if outfile[-3:]=='pha':
		header.append(('backfile', det+'.bkg', 'Name of corresponding background file (if any)'))
		header.append(('respfile', det+'.rsp', 'Name of corresponding RMF file (if any)'))
	else:
		header.append(('backfile', 'none', 'Name of corresponding background file (if any)'))
		header.append(('respfile', 'none', 'Name of corresponding RMF file (if any)'))
	header.append(('corrfile', 'none', 'Name of corresponding correction file (if any)'))
	header.append(('corrscal', 1., 'Correction scaling file'))
	header.append(('ancrfile', 'none', 'Name of corresponding ARF file (if any)'))
	header.append(('hduclass', 'OGIP', 'Format conforms to OGIP standard'))
	header.append(('hduclas1', 'SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)'))
	header.append(('hduclas2', 'TOTAL', 'Indicates gross data (source + background)'))
	header.append(('hduclas3', 'COUNT', 'Indicates data stored as counts'))
	header.append(('hduvers', '1.2.1', 'Version of HDUCLAS1 format in use'))
	header.append(('poisserr', True, 'Use Poisson Errors (T) or use STAT_ERR (F)'))
	header.append(('chantype', 'PHA', 'No corrections have been applied'))
	header.append(('detchans', 128, 'Total number of channels in each rate'))
	header.append(('hduclas4', 'TYPEI', 'PHA Type I (single) or II (mulitple spectra)'))
	
	header.comments['TTYPE1']='Label for column 1'
	header.comments['TFORM1']='2-byte INTERGER'
	header.comments['TTYPE2']='Label for column 2'
	header.comments['TFORM2']='8-byte DOUBLE'
	header.comments['TUNIT2']='Unit for colum 2'

	hdul = fits.HDUList([hdu0, hdu1])
	hdul.writeto(outfile)



def copy_rspI(bnname,det,outfile):
	shortyear=bnname[2:4]
	fullyear='20'+shortyear
	datadir=databasedir+'/'+fullyear+'/'+bnname+'/'
	rspfile=glob(datadir+'/'+'glg_cspec_'+det+'_'+bnname+'_v*.rsp')
	assert len(rspfile)==1, 'response file is missing for '+'glg_cspec_'+det+'_'+bnname+'_v*.rsp'
	rspfile=rspfile[0]
	os.system('cp '+rspfile+' '+outfile)
	

class GRB:
	def __init__(self,bnname):
		self.bnname=bnname
		resultdir=os.getcwd()+'/results/'
		self.resultdir=resultdir+'/'+bnname+'/'

		shortyear=self.bnname[2:4]
		fullyear='20'+shortyear
		self.datadir=databasedir+'/'+fullyear+'/'+self.bnname+'/'
		self.dataready=True
		for i in range(14):
			ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			if not len(ttefile)==1:
				self.dataready=False
			else:
				hdu=fits.open(ttefile[0])
				event=hdu['EVENTS'].data.field(0)
				if len(event)<10:
					self.dataready=False
		if self.dataready:
			if not os.path.exists(resultdir):
				os.makedirs(resultdir)
			if not os.path.exists(self.resultdir):
				os.makedirs(self.resultdir)
			self.baseresultdir=self.resultdir+'/base/'
			self.phaIresultdir=self.resultdir+'/phaI/'

			# determine GTI1 and GTI2
			GTI_t1=np.zeros(14)
			GTI_t2=np.zeros(14)
			for i in range(14):
				ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				GTI0_t1=time[0]
				GTI0_t2=time[-1]
				timeseq1=time[:-1]
				timeseq2=time[1:]
				deltime=timeseq2-timeseq1
				delindex=deltime>5 
				if len(timeseq1[delindex])>=1:
					GTItmp_t1=np.array(np.append([GTI0_t1],timeseq2[delindex]))
					GTItmp_t2=np.array(np.append(timeseq1[delindex],[GTI0_t2]))
					for kk in np.arange(len(GTItmp_t1)):
						if GTItmp_t1[kk]<=0.0 and GTItmp_t2[kk]>=0.0:
							GTI_t1[i]=GTItmp_t1[kk]
							GTI_t2[i]=GTItmp_t2[kk]
				else:
					GTI_t1[i]=GTI0_t1
					GTI_t2[i]=GTI0_t2
			self.GTI1=np.max(GTI_t1)
			self.GTI2=np.min(GTI_t2)

		
	def rawlc(self,viewt1=-50,viewt2=300,binwidth=0.1):		
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		assert viewt1<viewt2, self.bnname+': Inappropriate view times for rawlc!'
		if not os.path.exists(self.resultdir+'/'+'raw_lc.png'):
			#print('plotting raw_lc.png ...')
			tbins=np.arange(viewt1,viewt2+binwidth,binwidth)
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			for i in range(14):
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				ch=data.field(1)
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,2,125,126,127, notice 3-124
				goodindex=(ch>=ch1) & (ch<=ch2)  
				time=time[goodindex]
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				emin=emin[ch1:ch2+1]
				emax=ebound.field(2)
				emax=emax[ch1:ch2+1]
				histvalue, histbin =np.histogram(time,bins=tbins)
				plotrate=histvalue/binwidth
				plotrate=np.concatenate(([plotrate[0]],plotrate))
				axes[i//2,i%2].plot(histbin,plotrate,linestyle='steps')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=\
									axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].text(0.7,0.80,str(round(emin[0],1))+'-'+\
										str(round(emax[-1],1))+' keV',\
								transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',\
												 rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)
			plt.savefig(self.resultdir+'/raw_lc.png')
			plt.close()


	def base(self,baset1=-50,baset2=300,binwidth=0.1):
		self.baset1=np.max([self.GTI1,baset1])
		self.baset2=np.min([self.GTI2,baset2])
		self.binwidth=binwidth
		self.tbins=np.arange(self.baset1,self.baset2+self.binwidth,self.binwidth)
		assert self.baset1<self.baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.baseresultdir):
			os.makedirs(self.baseresultdir)
			expected_pvalue = norm_pvalue()
			f=h5py.File(self.baseresultdir+'/base.h5',mode='w')
			for i in range(14):
				grp=f.create_group(Det[i])
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+\
                     							self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])	
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				timedata=data.field(0)-trigtime
				chdata=data.field(1)
				for ch in range(128):
					time_selected=timedata[chdata==ch]
					histvalue, histbin=np.histogram(time_selected,bins=self.tbins)
					rate=histvalue/binwidth
					r.assign('rrate',rate) 
					r("y=matrix(rrate,nrow=1)")
					fillPeak_hwi=str(int(5/binwidth))
					fillPeak_int=str(int(len(rate)/10))
					r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,\
								 int ="+fillPeak_int+", method='fillPeaks')")
					r("bs=getBaseline(rbase)")
					r("cs=getCorrected(rbase)")
					bs=r('bs')[0]
					cs=r('cs')[0]
					corrections_index= (bs<0)
					bs[corrections_index]=0
					cs[corrections_index]=rate[corrections_index]
					f['/'+Det[i]+'/ch'+str(ch)]=np.array([rate,bs,cs])
			f.flush()
			f.close()

													
	def phaI(self,slicet1=0,slicet2=5):
		if not os.path.exists(self.phaIresultdir):
			os.makedirs(self.phaIresultdir)
		nslice=len(os.listdir(self.phaIresultdir))
		sliceresultdir=self.phaIresultdir+'/slice'+str(nslice)+'/'
		os.makedirs(sliceresultdir)
		fig, axes= plt.subplots(7,2,figsize=(32, 30),sharex=False,sharey=False)
		sliceindex= (self.tbins >=slicet1) & (self.tbins <=slicet2)
		valid_bins=np.sum(sliceindex)-1
		assert valid_bins>=1, self.bnname+': Inappropriate phaI slice time!'
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		for i in range(14):
			total_rate=np.zeros(128)
			bkg_rate=np.zeros(128)
			total_uncertainty=np.zeros(128)
			bkg_uncertainty=np.zeros(128)
			ttefile=glob(self.datadir+'/glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			hdu=fits.open(ttefile[0])
			ebound=hdu['EBOUNDS'].data
			emin=ebound.field(1)
			emax=ebound.field(2)
			energy_diff=emax-emin
			energy_bins=np.concatenate((emin,[emax[-1]]))
			for ch in range(128):
				base=f['/'+Det[i]+'/ch'+str(ch)][()][1]
				rate=f['/'+Det[i]+'/ch'+str(ch)][()][0]
				bkg=base[sliceindex[:-1]][:-1]
				total=rate[sliceindex[:-1]][:-1]
				bkg_rate[ch]=bkg.mean()
				total_rate[ch]=total.mean()
				exposure=len(bkg)*self.binwidth
				bkg_uncertainty[ch]=np.sqrt(bkg_rate[ch]/exposure)
				total_uncertainty[ch]=np.sqrt(total_rate[ch]/exposure)
			#plot both rate and bkg as count/s/keV
			write_phaI(bkg_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.bkg')
			write_phaI(total_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.pha')
			copy_rspI(self.bnname,Det[i],sliceresultdir+'/'+Det[i]+'.rsp')
			bkg_diff=bkg_rate/energy_diff
			total_diff=total_rate/energy_diff
			x=np.sqrt(emax*emin)
			axes[i//2,i%2].errorbar(x,bkg_diff,yerr=bkg_uncertainty/energy_diff,linestyle='None',color='blue')
			axes[i//2,i%2].errorbar(x,total_diff,yerr=total_uncertainty/energy_diff,linestyle='None',color='red')
			bkg_diff=np.concatenate(([bkg_diff[0]],bkg_diff))
			total_diff=np.concatenate(([total_diff[0]],total_diff))
			axes[i//2,i%2].plot(energy_bins,bkg_diff,linestyle='steps',color='blue')
			axes[i//2,i%2].plot(energy_bins,total_diff,linestyle='steps',color='red')
			axes[i//2,i%2].set_xscale('log')
			axes[i//2,i%2].set_yscale('log')
			axes[i//2,i%2].tick_params(labelsize=25)
			axes[i//2,i%2].text(0.85,0.85,Det[i],transform=\
										axes[i//2,i%2].transAxes,fontsize=25)
		fig.text(0.07, 0.5, 'Rate (count s$^{-1}$ keV$^{-1}$)', ha='center',\
							va='center', rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Energy (keV)', ha='center', va='center',\
															fontsize=30)	
		plt.savefig(sliceresultdir+'/PHA_rate_bkg.png')
		plt.close()
		f.close()


	def specanalyze(self,slicename):
		slicedir=self.phaIresultdir+'/'+slicename+'/'
		os.chdir(slicedir)
		# select the most bright two NaIs (in channels 6-118) 
		# and more bright one BGO (in channels 4-124):
		BGOtotal=np.zeros(2)
		NaItotal=np.zeros(12)
		for i in range(2):
			phahdu=fits.open(slicedir+'/'+BGO[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+BGO[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[4:125])
			plt.savefig(BGO[i]+'.png')
			plt.close()
			BGOtotal[i]=src[4:125].sum()
		for i in range(12):
			phahdu=fits.open(slicedir+'/'+NaI[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+NaI[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[6:118])
			plt.savefig(NaI[i]+'.png')
			plt.close()
			NaItotal[i]=src[6:118].sum()
		BGOindex=np.argsort(BGOtotal)
		NaIindex=np.argsort(NaItotal)
		brightdet=[BGO[BGOindex[-1]],NaI[NaIindex[-1]],NaI[NaIindex[-2]]]
		
		# use xspec

		alldatastr=' '.join(det1[i]+'.pha' for i in mask)
		#alldatastr=' '.join([det+'.pha' for det in brightdet])
		print(alldatastr)
		#input('--wait--')
		AllData(alldatastr)
		AllData.show()
		AllData.ignore('1-(l-1):**-8.0,800.0-**  l:**-200.0,40000.0-**')
		print(AllData.notice)
		
		Model('grbm')
		Fit.statMethod='pgstat'
		Fit.nIterations=1000
		Fit.query = "yes"
		Fit.perform()
		
		
		Fit.error('3.0 3')
		Fit.perform()		
		Plot.device='/null'
		Plot.xAxis='keV'
		Plot.yLog=True
		Plot('eeufspec')



		for i in range(1,1+l):
			energies=Plot.x(i)
			rates=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,rates,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('foldedspec.png')
		plt.close()
		Plot('eeufspec')

		for i in range(1,1+l):
			energies=Plot.x(i)
			ufspec=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,ufspec,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('eeufspec.png')
		plt.close()		

		par3=AllModels(1)(3)
		f = h5py.File(self.resultdir+"/data.h5", mode="w")
		epeak.append(par3.values[0])
		epeak_error_p.append(par3.error[0])
		epeak_error_n.append(par3.error[1])
		f = h5py.File("data.h5", mode="w")
		f["epeak"]=np.array(epeak)
		f["epeak_error_p"]=np.array(epeak_error_p)
		f["epeak_error_n"]=np.array(epeak_error_n)

		f.flush()
		f.close()		

		
		
	def removebase(self):
		os.system('rm -rf '+self.baseresultdir)

	def timeslice(self,lcbinwidth=0.05,gamma=1e-300):
		os.chdir(self.resultdir)
		det=['n3','n4']
		file = glob(self.datadir+'glg_tte_'+det[0]+'_'+self.bnname+'_v*.fit') 
		print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		data=hdu['events'].data['time']
		trigtime=hdu[0].header['trigtime']
		time=data-trigtime
		tte=time[(time>-10)&(time<50)]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		ax1.plot(plottime,plotrate,linestyle='steps',color='lightgreen')
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 

		l=len(edges)		
		for i in range(1,l-1):
			time_slice.append(edges[i])
		print(time_slice)
	

	def bbduration(self,lcbinwidth=0.05,gamma=1e-300):
		os.chdir(self.resultdir)
		det=['n3','n4']
		file = glob(self.datadir+'glg_tte_'+det[0]+'_'+self.bnname+'_v*.fit') 
		print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		data=hdu['events'].data['time']
		trigtime=hdu[0].header['trigtime']
		time=data-trigtime
		tte=time[(time>-10)&(time<50)]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		ax1.plot(plottime,plotrate,linestyle='steps',color='lightgreen')
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 
		ax1.plot(plottime,plotrate,linestyle='steps',color='b')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Count')
		l=len(edges)		

		
		x=[]
		dx=[]
		for i  in range(l-3):    
			s=(edges[i+1]+edges[i+2])/2
			z=(edges[i+2]-edges[i+1])/2
			x.append(s)
			dx.append(z)
		
		
		
		dy=[epeak_error_p,epeak_error_n]
		
		ax2.scatter(x,epeak,color='black', zorder=2,marker = '.',s=50.)    
		ax2.errorbar(x,epeak,xerr=dx,yerr=dy,zorder=1, fmt='o',color = '0.15',markersize=1e-50)
		
		ax2.set_ylim(0,600)
		ax2.set_ylabel('Epeak')
		plt.savefig('bbdurations.png')

	def hardness_ratio(self,lcbinwidth=0.1):
		os.chdir(self.resultdir)
		det=['n3','n4']
		file = glob(self.datadir+'glg_tte_'+det[0]+'_'+self.bnname+'_v*.fit') 
		print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		data=hdu['events'].data['time']
		trigtime=hdu[0].header['trigtime']
		time=data-trigtime
		tte=time[(time>-10)&(time<50)]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		ax1.plot(plottime,plotrate,linestyle='steps',color='lightgreen')
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 
		ax1.plot(plottime,plotrate,linestyle='steps',color='b')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Count')     
		x=[]
		dx=[]
		l=len(edges)	
		print('edges',edges)
		for i  in range(l-3):    
			s=(edges[i+1]+edges[i+2])/2
			z=(edges[i+2]-edges[i+1])/2
			x.append(s)
			dx.append(z)
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		
		cNetlo=np.array([f['/'+Det[5]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch3+1) ])
		cNethi=np.array([f['/'+Det[5]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch3,ch2+1) ])	
		totalNetlo=np.sum(cNetlo,axis=0)
		totalNethi=np.sum(cNethi,axis=0)
		hardness=totalNethi/totalNetlo
		hardness=np.concatenate(([hardness[0]],hardness))				
		edges=time_slice
		start=edges[:-1]
		stop=edges[1:]
		ever_rate=[]
		print('x',x)
		print(dx)
		for index,item in enumerate(start):
			t=np.where((self.tbins>=item)&(self.tbins<=stop[index]))[0]
			eva=hardness[t].mean()
			ever_rate.append(eva)		
		print('rate',ever_rate)

		ax2.scatter(x,ever_rate)
		ax2.errorbar(x,ever_rate,xerr=dx,zorder=1, fmt='o',color = '0.15',markersize=1e-50)
		ax2.set_ylim(0,2.1)
		ax2.set_ylabel('hardness ratio')
		plt.savefig('hardness_ratio.png')

for n in range(1,nl):
	os.chdir('/home/yao/Study/hardness-ratio')    
	bnname=name[n]
	print(bnname)
	number=trigger_name.tolist().index(bnname)
	a=float(t90_start_str[number])
	b=float(t90_str[number])+float(t90_start_str[number])
	print(a,b)
	Epeak=Flnc_Band_Epeak_str[number]
	mask=scat_detector_mask_str[number]
	det1=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
	mask = [m.start() for m in re.finditer('1', scat_detector_mask_str[number])]
	l=len(mask)
	print(mask)
	grb=GRB(bnname)
	grb.timeslice(lcbinwidth=0.05,gamma=1e-300)
	z=len(time_slice)
	print('time_slice:',time_slice)
	grb.rawlc(viewt1=-50,viewt2=300,binwidth=0.07)
	grb.base(baset1=-50,baset2=200,binwidth=0.07)

	#for i in range(z-1):
	#	grb.phaI(slicet1=time_slice[i],slicet2=time_slice[i+1])        
	#	grb.specanalyze('slice'+str(i))
	
	#print('epeak',epeak)
	grb.hardness_ratio(lcbinwidth=0.05)
	#grb.bbduration(lcbinwidth=0.05,gamma=1e-300)
	epeak=[]
	epeak_error_p=[]
	epeak_error_n=[]
	
