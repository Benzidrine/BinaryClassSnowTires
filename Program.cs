using System;
using SampleBinaryClassification.Model.DataModels;
using Microsoft.ML;

namespace consumeModelApp
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsumeModel();
        }

        public static void ConsumeModel()
        {
            // Load the model
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.WriteLine("Please enter the text to be judged for toxic sentiment:");
            
            Console.ForegroundColor = ConsoleColor.White;
            string InputLineToJudge = Console.ReadLine();

            // Use the code below to add input data
            var input = new ModelInput();
            input.SentimentText = InputLineToJudge;

            // Try model on sample data
            // True is toxic, false is non-toxic
            ModelOutput result = predEngine.Predict(input);

            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.WriteLine($"Prediction: {(Convert.ToBoolean(result.Prediction) ? "Toxic" : "Non Toxic")} sentiment");
        }
    }
}