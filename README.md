# MADE
Bayesian ANN, Bayesian spectroscopic mass estimator trained on APOKASC, and Bayesian isochrone pipeline for estimating distances and ages.

## CalcAstromDistGrid.py 	
Bayesian method for calculating distances for astrometric data (photometric data used for dust calcualtion).

## CalcSpectroPhotoAstromSeismoDistGrid.py 
Bayesian method to calculate masses, ages, metallicities, and distances for spectro-photo-astrom-seismo data.

## CalcApogeeTgasDR14Dist.py 	
Uses Bayesian method applied to astrometric data to estimate distances.

## CalcApogeeTgasDR14AgesDist.py
Uses Bayesian method applied to spectro-photo-astrom-seismo data, accounting for dust.

## CalcBNN.py
This module trains a Bayesian neural network and then applies it to make predictions in the 
case of unknown and known targets.
Initialize as

  bnn = CalcBNN()

## TrainApokascTgasMassesBNN.ipynb 
iPYTHON notebook creating a Bayesian spectrosopic mass estimator, using a Bayesian artificial neural network with APOKASC-TGAS DR14 training data.

## CalcApogeeTgasMasses.ipynb 	
iPYTHON notebook applying Bayesian spectroscopic mass estimator for estimating masses with whole of APOGEE-TGAS DR14.


