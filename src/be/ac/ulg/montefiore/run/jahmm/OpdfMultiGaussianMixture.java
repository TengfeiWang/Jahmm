/* jahmm package - v0.6.1 */

/*
  *  Copyright (c) 2004-2006, Jean-Marc Francois.
  *  Copyright (c) 2013, Aubry Cholleton:
  *  Adaptation of OpdfGaussianMixture to OpdfMultiGaussianMixture
 *
 *  This file is part of Jahmm.
 *  Jahmm is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Jahmm is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Jahmm; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 */

package be.ac.ulg.montefiore.run.jahmm;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;

import be.ac.ulg.montefiore.run.distributions.MultiGaussianDistribution;
import be.ac.ulg.montefiore.run.distributions.MultiGaussianMixtureDistribution;


/**
 * This class implements a mixture of monovariate gaussian distributions.
 *
 * @author Benjamin Chung (Creation)
 * @author Jean-Marc Francois (Adaptations / small fix)
 * @author Aubry Cholleton (Adaptations to multivariate gaussian mixtures models)
 */
public class OpdfMultiGaussianMixture implements Opdf<ObservationVector>
{
  private MultiGaussianMixtureDistribution distribution;
  
  
  /**
   * Creates a Gaussian mixture distribution. The mean values of the
   * distributions are evently distributed between 0 and 1 and each variance
   * is equal to 1.
   *
   * @param nbGaussians The number of gaussians that compose this mixture.
   */
  public OpdfMultiGaussianMixture(int nbGaussians)
  {
    distribution = new MultiGaussianMixtureDistribution(nbGaussians);
  }
  
  
  /**
   * Creates a Gaussian mixture distribution.  The mean and variance of
   * each distribution composing the mixture are given as arguments.
   *
   * @param means The mean values of the Gaussian distributions.
   * @param variances The variances of the Gaussian distributions.
   * @param proportions The mixing proportions. This array does not have to
   *             be normalized, but each element must be positive and the sum
   *             of its elements must be strictly positive.
   */
  public OpdfMultiGaussianMixture(double[][] means, double[][][] covariances,
      double[] proportions)
  {
    distribution = new MultiGaussianMixtureDistribution(means, covariances, 
        proportions);
  }
  
  
  public double probability(ObservationVector o)
  {
    return distribution.probability(o.value);
  }
  
  
  public ObservationVector generate()
  {
    return new ObservationVector(distribution.generate());
  }
  
  
  /**
   * Returns the number of distributions composing this mixture.
   * 
   * @return The number of distributions composing this mixture.
   */
  public int nbGaussians()
  {
    return distribution.nbGaussians();
  }
  
  public int dimension()
  {
    return distribution.dimension();
  }
  
  /**
   * Returns the mixing proportions of each gaussian distribution.
   * 
   * @return A (copy of) array giving the distributions' proportion.
   */
  public double[] proportions()
  {
    return distribution.proportions();
  }

  
  /**
   * Returns the mean value of each distribution composing this mixture.
   * 
   * @return A copy of the means array.
   */
  public double[][] means()
  {   
    double[][] means = new double[nbGaussians()][dimension()];
    MultiGaussianDistribution[] distributions = distribution.distributions(); 

    for (int i = 0; i < distributions.length; i++)
      means[i] = distributions[i].mean();
    
    return means;
  }
  
  
  /**
   * Returns the mean value of each distribution composing this mixture.
   * 
   * @return A copy of the means array.
   */
  public double[][][] covariances()
  {
    double[][][] covariances = new double[nbGaussians()][dimension()][dimension()];
    GaussianDistribution[] distributions = distribution.distributions(); 

    for (int i = 0; i < distributions.length; i++)
      covariances[i] = distributions[i].covariance();
    
    return covariances;
  }
  
  
  /**
   * Fits this observation distribution function to a (non
   * empty) set of observations.  This method performs one iteration of
   * an expectation-maximisation algorithm.
   *
   * @param oa A set of observations compatible with this function.
   */
  public void fit(ObservationVector... oa)
  {
    fit(Arrays.asList(oa));
  }
  
  
  /**
   * Fits this observation distribution function to a (non
   * empty) set of observations.  This method performs one iteration of
   * an expectation-maximisation algorithm.
   *
   * @param co A set of observations compatible with this function.
   */
  public void fit(Collection<? extends ObservationVector> co)
  {
    if (co.isEmpty())
      throw new IllegalArgumentException("Empty observation set");

    double[] weights = new double[co.size()];
    Arrays.fill(weights, 1. / co.size());
    
    fit(co, weights);
  }
  
  
  /**
   * Fits this observation distribution function to a (non
   * empty) weighted set of observations.  This method performs one iteration
   * of an expectation-maximisation algorithm.  Equations (53) and (54)
   * of Rabiner's <i>A Tutorial on Hidden Markov Models and Selected 
   * Applications in Speech Recognition</i> explain how the weights can be
   * used.
   *
   * @param o A set of observations compatible with this function.
   * @param weights The weights associated to the observations.
   */
  public void fit(ObservationVector[] o, double[] weights)
  {
    fit(Arrays.asList(o), weights);
  }
  
  
  /**
   * Fits this observation distribution function to a (non
   * empty) weighted set of observations.  This method performs one iteration
   * of an expectation-maximisation algorithm.  Equations (53) and (54)
   * of Rabiner's <i>A Tutorial on Hidden Markov Models and Selected 
   * Applications in Speech Recognition</i> explain how the weights can be
   * used.
   *
   * @param co A set of observations compatible with this function.
   * @param weights The weights associated to the observations.
   */
  public void fit(Collection<? extends ObservationVector> co,
      double[] weights)
  {
    if (co.isEmpty() || co.size() != weights.length)
      throw new IllegalArgumentException();
    
    ObservationVector[] o = co.toArray(new ObservationVector[co.size()]);
    
    double[][] delta = getDelta(o);
    double[] newMixingProportions = 
      computeNewMixingProportions(delta, o, weights);
    double[][] newMeans = computeNewMeans(delta, o, weights);
    double[][][] newCovariances = computeNewCovariances(delta, o, weights);
    
    distribution = new GaussianMixtureDistribution(newMeans, newCovariances,
        newMixingProportions);
  }
  
  
  /* 
   * Computes the relative weight of each observation for each distribution.
   */
  private double[][] getDelta(ObservationVector[] o)
  {
    double[][] delta = new double[distribution.nbGaussians()][o.length];
    
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      double[] proportions = distribution.proportions();
      MultiGaussianDistribution[] distributions =
        distribution.distributions();
      
      for (int t = 0; t < o.length; t++)
        delta[i][t] = proportions[i] *
        distributions[i].probability(o[t].value) / probability(o[t]);
    }
      
    return delta;
  }
  
  
  /*
   * Estimates new mixing proportions given delta.
   */
  private double[] computeNewMixingProportions(double[][] delta, 
      ObservationVector[] o, double[] weights)
  {
    double[] num = new double[distribution.nbGaussians()];
    double sum = 0.0;
    
    Arrays.fill(num, 0.0);
    
    for (int i = 0; i < distribution.nbGaussians(); i++)
      for (int t = 0; t < weights.length; t++) {
        num[i] += weights[t] * delta[i][t];
        sum += weights[t] * delta[i][t];
      }
    
    double[] newMixingProportions = new double[distribution.nbGaussians()];
    for (int i = 0; i < distribution.nbGaussians(); i++) 
      newMixingProportions[i] = num[i]/sum;
    
    return newMixingProportions;
  }
  
  
  /*
   * Estimates new mean values of each Gaussian given delta.
   */
  private double[][] computeNewMeans(double[][] delta, ObservationVector[] o,
      double[] weights)
  {
    double[][] num = new double[distribution.nbGaussians()][distribution.dimension()];
    double[] sum = new double[distribution.nbGaussians()];
    
    Arrays.fill(num, 0.0);
    Arrays.fill(sum, 0.0);
    
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      for (int t = 0; t < o.length; t++) {
        for (int d = 0; d < distribution.dimension(); d++) {
          num[i][d] += weights[t] * delta[i][t] * o[t][d].value;
        }
        sum[i] += weights[t] * delta[i][t];
      }
    }
    
    double[][] newMeans = new double[distribution.nbGaussians()][distribution.dimension()];
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      for (int d = 0; d < distribution.dimension(); d++) {
        newMeans[i][d] = num[i][d] / sum[i];
      }
    }

    return newMeans;
  }
  
  
  /*
   * Estimates new variance values of each Gaussian given delta.
   */
  private double[][][] computeNewCovariances(double[][] delta, ObservationVector[] o,
      double[] weights)
  {
    double[][] num = new double[distribution.nbGaussians()][distribution.dimension()];
    double[] sum = new double[distribution.nbGaussians()];
    
    Arrays.fill(num, 0.);
    Arrays.fill(sum, 0.);
    
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      MultiGaussianDistribution[] distributions = distribution.distributions();
      
      for (int t = 0; t < o.length; t++) {
        for (int x = 0; x < distribution.dimension(); x++) {
          num[i][x] += weights[t] * delta[i][t] *
          (o[t][x].value - distributions[i].mean()) *
          (o[t][x].value - distributions[i].mean());
        }
        sum[i] += weights[t] * delta[i][t];
      }
    }
    
    double[][][] newCovariances = new double[distribution.nbGaussians()][distribution.dimension()][distribution.dimension()];
    Arrays.fill(newCovariances, 0.);
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      for (int x = 0; x < distribution.dimension(); x++) {
        newCoariances[i][x][x] = num[i][x] / sum[i];
      }
    }
    
    return newCovariances;
  }
  
  
  public OpdfGaussianMixture clone()
  {
    try {
      return (OpdfGaussianMixture) super.clone();
    } catch(CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
  }
  
  
  public String toString() {
    return toString(NumberFormat.getInstance());
  }
  
  
  public String toString(NumberFormat numberFormat)
  {
    String s = "Gaussian mixture distribution --- ";
    
    double[] proportions = proportions();
    double[] means = means();
    double[] covariances = covariances();
    
    for (int i = 0; i < distribution.nbGaussians(); i++) {
      s += "Gaussian " + (i+1) + ":\n";
      s += "\tMixing Prop = " + numberFormat.format(proportions[i]) +
      "\n";
      s += "\tMean = " + numberFormat.format(means[i]) + "\n";
      s += "\tCovariance matrix = " + numberFormat.format(covariances[i]) + "\n";
    }
    
    return s;
  }    


  private static final long serialVersionUID = 1L;
}
