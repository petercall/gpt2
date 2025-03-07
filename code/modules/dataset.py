from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, context_length, slide_length):
        super(TextDataset, self).__init__()
        self.data = text
        self.context_length = context_length
        self.slide_length = slide_length

    def __len__(self):
        return (len(self.data)//self.slide_length) - 1

    def __getitem__(self, i):
        start = i*self.slide_length
        return self.data[start : start+self.context_length], self.data[start+1 : start+self.context_length+1]