package naivebayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

@SuppressWarnings("deprecation")
public class NavieBayesian {
	static String model;
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Instances instances;
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		while(true)
		{
			System.out.println("*******************************Menu***************************************");
			System.out.println("1. Learn a Naïve Bayesian classifier from data. \n2. Load and test accuracy of a naïve Bayesian classifier. \n3. Apply a naïve Bayesian classifier to new cases. \n4. Quit.");
			System.out.println("**************************************************************************");
			System.out.println("Please enter your choice:");
			int ch=Integer.parseInt(br.readLine());
			switch(ch)
			{
			case 1:
				System.out.println("Enter the full path to the directory in data file is present:");
				String path=br.readLine();
				System.out.println("Enter the name of the file without the extension:");
				String file_name=br.readLine();
				String full_path=path+"/"+file_name+".arff";
				ArffLoader loader = new ArffLoader();
				loader.setFile(new File(full_path));
				Instances data = loader.getStructure();
				 if (data.classIndex() == -1)
					   data.setClassIndex(data.numAttributes() - 1);
				 NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
				 nb.buildClassifier(data);
				 Instance current;
				 while ((current = loader.getNextInstance(data)) != null)
					   nb.updateClassifier(current);
				 weka.core.SerializationHelper.write(path+"/"+file_name+".bin", nb);
				 System.out.println("File Saved at "+path+"/"+file_name+".bin!");
				 break;
			case 2:
				System.out.println("Enter the full path of the model file:");
				model=br.readLine();
				NaiveBayes cls = (NaiveBayes) weka.core.SerializationHelper.read(model);
				System.out.println("Enter the full path of the test file:");
				String test_file=br.readLine();
				ArffLoader loader1 = new ArffLoader();
				loader1.setFile(new File(test_file));
				Instances data1 = loader1.getDataSet();
				 if (data1.classIndex() == -1)
					   data1.setClassIndex(data1.numAttributes() - 1);
				 Evaluation eval_train = new Evaluation(data1);
				 eval_train.evaluateModel(cls,data1);
				 System.out.println(eval_train.toMatrixString("\nConfusion Matrix:\n========================\n"));
				 break;
			case 3:
				while(true)
				{
					NaiveBayes nb1 = (NaiveBayes) weka.core.SerializationHelper.read("weather.nominal.bin");
					System.out.println("1.Enter a new case interactively.\n2.Quit.");
					int sub_ch=Integer.parseInt(br.readLine());
					if(sub_ch!=2)
					{
					System.out.println("Please select a value from {sunny, overcast, rainy} for outlook:");
					String a1=br.readLine();
					System.out.println("Please select a value from {hot, mild, cool} for temperature");
					String a2=br.readLine();
					System.out.println("Please select a value from {high, normal} for  humidity");
					String a3=br.readLine();
					System.out.println("Please select a value from {TRUE, FALSE} for  windy");
					String a4=br.readLine();
					a4=a4.toUpperCase();
					FastVector<String> OutlookNominalVal = new FastVector<String>(3);
					OutlookNominalVal.addElement("sunny");
					OutlookNominalVal.addElement("overcast");
					OutlookNominalVal.addElement("rainy");
					FastVector<String> temperatureNominalVal = new FastVector<String>(3);
					temperatureNominalVal.addElement("hot");
					temperatureNominalVal.addElement("mild");
					temperatureNominalVal.addElement("cool");
					FastVector<String> humidityNominalVal = new FastVector<String>(2);
					humidityNominalVal.addElement("high");
					humidityNominalVal.addElement("normal");
					FastVector<String> WindyNominalVal = new FastVector<String>(2);
					WindyNominalVal.addElement("TRUE");
					WindyNominalVal.addElement("FALSE");
					FastVector<String> playNominalVal = new FastVector<String>(2);
					playNominalVal.addElement("yes");
					playNominalVal.addElement("no");
					Attribute attr1 = new Attribute("outlook",OutlookNominalVal);
				    Attribute attr2 = new Attribute("temperature",temperatureNominalVal);
				    Attribute attr3 = new Attribute("humidity",humidityNominalVal);
				    Attribute attr4 = new Attribute("windy",WindyNominalVal);
				    Attribute attr5 = new Attribute("play",playNominalVal);
				    FastVector<Attribute> fvWekaAttributes = new FastVector<Attribute>(5);
				    fvWekaAttributes.addElement(attr1);
				    fvWekaAttributes.addElement(attr2);
				    fvWekaAttributes.addElement(attr3);
				    fvWekaAttributes.addElement(attr4);
				    fvWekaAttributes.addElement(attr5);
				    instances = new Instances("Test relation", fvWekaAttributes,1);
				    instances.setClassIndex(4);
			        DenseInstance instance = new DenseInstance(5);
			        instance.setValue(attr1, a1);
			        instance.setValue(attr2, a2);
			        instance.setValue(attr3, a3);
			        instance.setValue(attr4, a4);
			        instances.add(instance);
				    double myValue = nb1.classifyInstance(instances.lastInstance());
				    String prediction = instances.classAttribute().value((int) myValue);
				    System.out.println("The predicted value of the play is:" + prediction);
				    System.out.println("*****************************************************************");				    
					}
					else
					{
						System.out.println("You chose to stop entering values");
						break;
					}
				}
				break;
			case 4:
				System.out.println("You chose to quit!The program will now terminate.");
				System.exit(0);
				break;
			default:
				System.out.println("Wrong Choice! Please enter the choice again!");
				
			}
		}
	}
}
