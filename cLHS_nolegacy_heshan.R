library(spcosa)
library(sp)

# Source annealing functions
source('Functions4SSA.R')

# Read grid with covariates
# for raffelson dataset
#grd<-read.csv(file="./data/envdata_raf.csv",header=TRUE)

# for heshan dataset
grd<-read.csv(file="./data/envdata_hs.csv",header=TRUE)

head(grd)

# A problem is that the spacing of the grid is not exactly constant, which causes problems with typecasting the SpatialPointsDataFrame to a SpatialPixelsDataFrame.
# This problem is solved by rounding the coordinates.

cellsize = 10
grd$x <- cellsize * round(grd$x/cellsize, 0)
grd$y <- cellsize * round(grd$y/cellsize, 0)


# In which columns are the coordinates and covariates?
# for raffelson dataset 
#col.xy <- c(1,2)
#col.cov <- c(3,4,5,6,7,8,9)

# for heshan dataset 
col.xy <- c(1,2)
col.cov <- c(3,4,5,6)

# Compute population correlation matrix of covariates
R<-cor(grd[,col.cov])

# Typecast grd to SpatialPixelsDataFrame
grid <- SpatialPixelsDataFrame(
    points = grd[,col.xy],
    data   = grd[,col.cov]
)

# Select legacy sample
#set.seed(272)
#ids <- sample.int(nrow(grd),10)
#legacy <- SpatialPoints(
#  coords=grd[ids,col.xy]
#)

# Select spatial infill sample which is used as initial sample in annealing
set.seed(314)
samplesize<-50 #number of additional points
#ntot <- samplesize+length(legacy)
ntot <- samplesize

#myStrata <- stratify(grid, nStrata=ntot, priorPoints=legacy, equalArea=FALSE, nTry=1)
myStrata <- stratify(grid, nStrata=ntot, equalArea=FALSE, nTry=1)
mySample <- spsample(myStrata)
#plot(myStrata, mySample)

# Select the new points from mySample
#ids <- which(mySample@isPriorPoint==F)
mySample <- as(mySample, "SpatialPoints")
#mySample <- mySample[ids,]

# Compute lower bounds of marginal strata
by=1/ntot
probs<-seq(from=0,to=1,by=by)
last<-length(probs)
lb<-matrix(ncol=length(col.cov),nrow=length(probs)-1)
for (i in 1:length(col.cov)) {
  q<-quantile(grd[,col.cov[i]],probs=probs)
  lb[,i]<-q[-last]
}

#set relative weight of O1 for computing the LHS criterion (O1 is for coverage of marginal strata of covariates); 1-W01  is the relative weight for O3 (for correlation)
wO1<-0.5

#now start the annealing
system.time(
annealingResult <- anneal.cLHS(
    d = mySample,
    g = grid,
    #legacy = legacy,
    lb = lb,
    wO1 = wO1,
    R = R,
    initialTemperature = 2,
    coolingRate = 0.9,
    maxAccepted = 5*length(mySample),
    maxPermuted = 5*length(mySample),
    maxNoChange=10,
    verbose = "TRUE"
    )
)

#save(annealingResult,file="LHSample_10_20(0.5)_raf.Rdata")
#load(file="LHSample_10_20(0.5)_raf.Rdata")
save(annealingResult,file="LHSample_0_(0.5)_raf.Rdata")
load(file="LHSample_0_(0.5)_raf.Rdata")

# show the location of samples
annealingResult$optSample@coords
write.csv(annealingResult$optSample@coords, file=paste('./data/result/test5/cLHS_0+',samplesize,'.csv', sep=''), row.names = F, quote = F)

# optSample<-as(annealingResult$optSample, "data.frame")
# Eall<-annealingResult$Criterion
# 
# #Plot the selected points on top of one of the covariates
# legacy <- as(legacy, "data.frame")
# #pdf(file = "LHSample_50(05)_raf_elevation.pdf", width = 7, height = 7)
# ggplot(data=grd) +
#   geom_tile(mapping = aes(x = x, y = y, fill = env2)) +  
#   geom_point(data = optSample, mapping = aes(x = x, y = y), colour = "black") +
#   #geom_point(data = legacy, mapping = aes(x = x, y = y), colour = "red") +
#   scale_x_continuous(name = "Easting (km)") +
#   scale_y_continuous(name = "Northing (km)") +    
#   scale_fill_gradient(name="elevation", low = "darkblue", high = "red") +
#   coord_fixed()
# #dev.off()
# 
# #Make scatter plots
# coordinates(optSample)<-~x+y
# optSample <- over(optSample, grid)
# 
# coordinates(legacy)<-~x+y
# legacy <- over(legacy, grid)
# 
# #pdf(file = "LHSample_50(0.5)_raf_elevation_vs_twi.pdf", width = 7, height = 7)
# ggplot(data=grd) +
#   geom_point(mapping = aes(x = env2, y = env6), colour = "black", size=1, alpha=0.5) +
#   geom_point(data=optSample, mapping = aes(x = env2, y = env6), colour = "red", size=2) +
#   #geom_point(data=as.data.frame(legacy), mapping = aes(x = env2, y = env6), colour = "green", size=2) +
#   scale_x_continuous(name = "elevation") +
#   scale_y_continuous(name = "twi")
# dev.off()
# 
# #Plot O1
# index<-seq(1:samplesize)
# countsdf<-as.data.frame(counts)
# names(countsdf)<-names(grd[c(-1,-2)])
# countsdf<-countsdf-1
# sum(countsdf)
# library(reshape)
# countslf<-melt(countsdf)
# countslf$index<-rep(index,times=4)
# pdf(file = "O1_LHSample_50(05).pdf", width = 8, height = 4)
#   ggplot(countslf) +
#   geom_point(mapping = aes(x=index,y = value), colour = "black",size=1) +
#   facet_wrap(~variable) +
#   scale_x_continuous(name = "Index") +
#   scale_y_continuous(name = "Difference",breaks=c(-1,0,1))
# dev.off()