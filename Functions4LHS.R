#define function for computing the criterion to be minimized; arguments of function are d, g and model
#d: SpatialPoints
#g: SpatialPixelsDataFrame
#model: semivariogram (van package gstat)
getCriterion<-function(d,g,lb,wO1,R)  {
  #determine values of covariates at locations in d
    d <- SpatialPointsDataFrame(
        coords = d,
        data = over(d,grid)                
    )
    
  
  #Determine in which stratum the sampling locations are
  stratum<-matrix(nrow=length(d),ncol=ncol(d))
  for ( i in 1:ncol(d) ) {
    stratum[,i]<-findInterval(as.data.frame(d[,i])[,1],lb[,i])
  }
  
  #count number of points in marginal strata
  counts<-matrix(nrow=nrow(lb),ncol=ncol(d))
  for (i in 1:nrow(lb)) {
    counts[i,]<-apply(stratum, MARGIN=2, function(x,i) sum(x==i), i=i)
  }
  O1<-sum(abs(counts-length(d)/nrow(lb)))
  
  #compute sum of absolute differences of correlations
  r<-cor(as.data.frame(d)[1:4])
  O3<-sum(abs(R-r))
  
  #compute LHS criterion
  E<-wO1*O1+(1-wO1)*O3
  
  # return result
  c(E = E, O1 = O1, O3=O3)
}



#define function for generating a series of samples to be evaluated
permute<-function(d, g)  {
  # extract coordinates of observation points 'd' and grid cells 'g'
  s_d <- coordinates(d)
  s_g <- coordinates(g)
  
  # randomly select one location in 'd'
  i_d <- sample(x = seq_len(nrow(s_d)), size = 1)
  
  # compute squared Euclidean distances 'd2' between the selected location and all grid cells
  d2 <- (s_g[, 1] - s_d[i_d, 1])^2 +
    (s_g[, 2] - s_d[i_d, 2])^2
  
  # randomly select a grid cell with a probability inverse to squared distance (p ~ 1/distance^2)
  i_g <- sample(x = seq_len(nrow(s_g)), size = 1, prob = 1/(d2 + 1))
  
  # replace randomly selected location in actual sample (s_d[i_d, ]) by a new location within the randomly selected grid cell (g[i_g, ])
  gridTopology <- as(getGridTopology(g), "data.frame")
  s_d[i_d, ] <- s_g[i_g, ] + runif(n = 2, min = -0.5, max = 0.5) * gridTopology$cellsize
  
  # return result
  SpatialPoints(coords = s_d)
}

#define annealing function; default values are given for the arguments initialTemperature, coolingRate et cetera

anneal<-function(d, g, lb, wO1, R,
                 initialTemperature = 1, coolingRate = 0.9, maxAccepted = 10 * nrow(coordinates(d)),
                 maxPermuted=10* nrow(coordinates(d)),maxNoChange=nrow(coordinates(d)),verbose = getOption("verbose")) {
  
  # set initial temperature
  T <- initialTemperature
  
  # compute the criterion
  criterion <- getCriterion(d, g, lb, wO1,R)
  
  # store criterion
  criterion_prv <- criterion
  
  # Define structure for storing time series of criterion
  Eall<-NULL

  # initialize number of zero changes of objective function
  nNoChange <-0
  
  # start cooling loop
  repeat{
    
    # initialize number of accepted configurations
    nAccepted <- 0
    
    # initialize number of permuted configurations
    nPermuted <- 0
    
    # initialize number of improved configurations
    nImproved <- 0
            
    # start permutation loop
    repeat {
      
      # increase the number of permutations
      nPermuted <- nPermuted + 1
      
      # propose new sample by making use of function permute
      d_p <- permute(d, g)
      
      # compute the criterion of this new sample by using function getCriterion
      criterion_p <- getCriterion(d_p, g, lb, wO1, R)
      
      # accept/reject proposal by means of Metropolis criterion
      dE <- criterion_p["E"] - criterion["E"]
      if (dE < 0) {
          nImproved <- nImproved + 1
          p <- 1 # always accept improvements
      } else {
          p <- exp(-dE / T) # use Boltzmann to judge if deteriorations should be accepted
        }
      u <- runif(n = 1) # draw uniform deviate
      if (u < p) { # accept proposal
          nAccepted <- nAccepted + 1
          d <- d_p
          criterion <- criterion_p
      }
      
      
      # are conditions met to lower temperature?
      lowerTemperature <- (nPermuted == maxPermuted) |
        (nAccepted == maxAccepted)
      if (lowerTemperature) {
        if (nImproved==0)
          {nNoChange<-nNoChange+1}
        else
          {nNoChange<-0}
        Eall<-rbind(Eall,criterion)
        break  
      }
    }
    
    if (verbose) {
      cat(
        format(Sys.time()), "|",
        sprintf("T = %e  E = %e  permuted = %d  accepted = %d  improved = %d  acceptance rate = %f  \n",
                T, criterion["E"], nPermuted, nAccepted, nImproved, nAccepted / nPermuted)
      )
    }
    
    # check on convergence
    if (nAccepted == 0L | nNoChange == maxNoChange) {
      break
    }
    criterion_prv <- criterion
    
    # lower temperature
    T <- coolingRate * T
  }
  
  # return result
  list(
    optSample=d,Criterion=Eall
  )
}
