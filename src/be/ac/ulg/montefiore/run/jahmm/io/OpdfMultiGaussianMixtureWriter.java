package be.ac.ulg.montefiore.run.jahmm.io;

import java.io.IOException;
import java.io.Writer;

import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussianMixture;


/**
 * This class implements a {@link OpdfMultiGaussianMixture} writer.  It is
 * compatible with the {@link OpdfMultiGaussianMixtureReader} class.
 */
public class OpdfMultiGaussianMixtureWriter
extends OpdfWriter<OpdfMultiGaussianMixture>
{
  public void write(Writer writer, OpdfMultiGaussianMixture opdf)
  throws IOException
  {
    writer.write("MultiGaussianMixtureOPDF [ ");

    write(writer, opdf.proportions());

    writer.write(" [");
    for (double[] line : opdf.means()) {
      writer.write(" ");
      write(writer, line);
    }
    writer.write(" ]");

    writer.write(" [");
    for (double[][] matrix : opdf.covariances()) {
      writer.write(" [");
      for (double[] line : matrix) {
        writer.write(" ");
        write(writer, line);
      }
      writer.write(" ]");
    }
    writer.write(" ] ");

    writer.write(" ]");
  }
}