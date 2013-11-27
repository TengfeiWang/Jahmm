package be.ac.ulg.montefiore.run.jahmm.io;

import java.io.IOException;
import java.io.StreamTokenizer;

import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussian;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussianMixture;


/**
 * This class implements a {@link OpdfMultiGaussianMixture} reader.  The syntax of the
 * distribution description is the following.
 */
public class OpdfMultiGaussianMixtureReader
extends OpdfReader<OpdfMultiGaussianMixture>
{
  String keyword()
  {
    return "MultiGaussianMixtureOPDF";
  }

  
  public OpdfMultiGaussianMixture read(StreamTokenizer st)
  throws IOException, FileFormatException {
    HmmReader.readWords(st, keyword(), "[");

    double[] proportions = OpdfReader.read(st, -1);

    double[][] means = new double[proportions.length][];
    
    HmmReader.readWords(st, "[");
    for (int l = 0; l < means.length; l++)
      means[l] = OpdfReader.read(st, -1);
    HmmReader.readWords(st, "]");

    double[][][] covariances = new double[proportions.length][means[0].length][means[0].length];
    
    HmmReader.readWords(st, "[");
    for (int l = 0; l < covariances.length; l++) {
      HmmReader.readWords(st, "[");
      for (int m = 0; m < covariances[0].length; m++)
        covariances[l][m] = OpdfReader.read(st, covariances[0].length);
      HmmReader.readWords(st, "]");
    }
    HmmReader.readWords(st, "]");
    
    HmmReader.readWords(st, "]");
    
    return new OpdfMultiGaussianMixture(means, covariances,
        proportions);
  }
}