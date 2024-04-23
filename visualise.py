from visualise_masks import visualise_segmentation
from datasets import CONSEP
from torch.utils.data import DataLoader

def visualise():
    dataset_test = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/test', mode='val')
    dataloader = DataLoader(dataset_test, batch_size=32, num_workers=10, pin_memory=True)
    
    visualise_segmentation('TwoEncodersOneDecoder_consep.pth', 'Graphs', dataloader)

if __name__ == '__main__':
    visualise()