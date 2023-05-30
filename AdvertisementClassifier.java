import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.io.File;

public class AdvertisementClassifier {
    public static void main(String[] args) {
        try {
            // Load dataset
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("data.arff"));
            Instances dataset = loader.getDataSet();
            
            // Set class attribute index
            dataset.setClassIndex(3);
            
            // Build Decision Tree model
            J48 tree = new J48();
            tree.setOptions(new String[] {"-U"}); // Menambahkan opsi -U untuk menampilkan gain
            tree.buildClassifier(dataset);
            
            // Print Decision Tree model
            System.out.println(tree);

            // Perform cross-validation
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(tree, dataset, 10, new java.util.Random(1));
            
            // Print accuracy
            System.out.println("Accuracy: " + eval.pctCorrect());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
// A Apakah pengguna adalah individu yang berpenghasilan tinggi?
// B Apakah pengguna tertarik dengan iklan contoh jam tangan?
// C Apakah pengguna pernah membeli item sebelumnya?
// P Akankah pengguna membeli barang yang dari iklan seperti pertanyaan B jam tangan?


// J48 pruned tree
// ------------------
// A = T
// |   B = T: T (6.0/2.0)
// |   B = F: F (2.0)
// A = F: F (2.0)
// hasil yang diberikan menunjukkan bahwa model Decision Tree memprediksi bahwa jika A = T dan B = T,
// maka hasilnya adalah T (User akan membeli jam tangan). 
// Jika A = T dan B = F, maka hasilnya adalah F (User tidak akan membeli jam tangan). 
// Jika A = F, maka hasilnya adalah F (User tidak akan membeli jam tangan).