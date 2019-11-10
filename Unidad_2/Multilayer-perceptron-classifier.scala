// $example on$
//Loading required packages and APIs
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example for Multilayer Perceptron Classification.
 */
 ///Creating a Spark session
 val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    //Load the input data in libsvm format.
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")

    // Split the data into train and test
    //Preparing the training and testing set
    //Prepare the train and test set: training => 60%, test => 40% and seed => 1234L
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // train the model
    val model = trainer.fit(train)
