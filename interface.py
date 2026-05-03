#replace MyCustomModel with the name of your model
from model import ImageClassifier as TheModel

from train import train_model as the_trainer

from predict import predict_images as the_predictor

from predict import inferloader as the_inferloader

from dataset import AIRealDataset as TheDataset

from dataset import get_dataloader as the_dataloader

from config import batch_size as the_batch_size

from config import epochs as total_epochs

from config import sample_data as the_data_dir