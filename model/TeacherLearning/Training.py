import pandas as pd
import tensorflow as tf
import transformers
from TeacherLearning import TeacherLearningModel
import Utils
import Dataset
import TrainingModel

# Function to load custom dataset with English and Mongolian captions
def loadCustomDataset(csvFile):
    # Load your custom CSV with image_id, English (eng), and Mongolian (mon)
    df = pd.read_csv(csvFile)
    
    # Example CSV structure: image_id, eng, mon
    # Convert to HuggingFace datasets format
    dataset = Dataset.from_pandas(df[['image_id', 'eng', 'mon']])
    return dataset

# Function to load CLIP embeddings
def loadEmbeddings():
    # Load precomputed English embeddings from the CSV
    df = pd.read_csv('english_embeddings.csv')

    # Assuming the embeddings are stored in a column like 'eng_embedding'
    embeddings = df['eng_embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).tolist()
    return embeddings

def singleGPUTraining():
    numValidationSamples = 5000
    stepsPerEpoch, lr = 1000, 0.00001
    gradAccumSteps, batchSize = 1, 256
    numTrainSteps, numWarmupSteps = 99999999, 1000

    modelBase = 'xlm-roberta-large'
    tokenizerBase = 'xlm-roberta-large'
    imageBase = "Vit-B-32"
    modelName = '{}-{}'.format(modelBase, imageBase)

    startWeights = None
    targetCaptions = loadCustomDataset('your_flickr30k_dataset.csv')  # Load your custom dataset
    trainEmbeddings, valEmbeddings, imageEncoderDimensions = loadEmbeddings()  # Load precomputed embeddings

    def createOptimizerFunc():
        optimizer, schedule = transformers.optimization_tf.create_optimizer(lr, numTrainSteps, numWarmupSteps)
        if gradAccumSteps <= 1:
            return optimizer
        else:
            return Utils.GradientAccumulator(optimizer, gradAccumSteps)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerBase)
    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, imageEncoderDimensions[-1])

    if startWeights is not None:
        model.load_weights(startWeights)
    model.compile(createOptimizerFunc(), 'mse', metrics=['mae', 'cosine_similarity'])

    # Create datasets
    trainDataset, valDataset = Dataset.createTrainingAndValidationDataset(trainEmbeddings, valEmbeddings, batchSize,
                                                                          tokenizer, targetCaptions=targetCaptions,
                                                                          encoderDims=imageEncoderDimensions)

    if gradAccumSteps > 1:  # In order to make fair logging on Wandb
        stepsPerEpoch *= gradAccumSteps

    model.fit(trainDataset, epochs=1000, steps_per_epoch=stepsPerEpoch,
              validation_data=valDataset,
              callbacks=[
                  Utils.CustomSaveCallBack(modelName, saveInterval=5, firstSavePoint=5),
              ])

if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        singleGPUTraining()
